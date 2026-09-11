#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

# MIT License
#
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Minimap2 Stage Analyzer

Profiles the minimap2 pipeline by classifying CPU and GPU functions into
6 stages: seeding, chaining, alignment, I/O, gpu_other, and other.
Uses up to 3 profiling backends (each independently skippable):

  - Intel VTune (ITT): CPU hotspot functions with time attribution
  - rocprofv3: GPU kernels, HIP API calls, ROCTX markers
  - Trace logs: Lightweight timing from spdlog trace output

Pipeline stage classification is auto-discovered by scanning minimap2
source code for MM_PROFILE_SEED/CHAIN/ALIGN/IO/GPU macros at runtime.

Arguments:
  -m, --minimap2 PATH    Path to minimap2 (auto-detected if omitted)
  -o, --output DIR       Output directory for results
  -f, --format FMT       Output format: text or json (default: text)
  --no-vtune             Skip VTune profiling
  --no-rocprof           Skip rocprofv3 GPU profiling
  --no-trace             Skip trace log timing
  --reuse-data           Reuse existing profiles from output dir (no re-run)
  --min-duration US      Filter functions below this duration in us (default: 0)
  mm_args                Minimap2 arguments (placed after -- separator)

Prerequisites:
  - Intel VTune (vtune) on PATH -- skipped gracefully if absent
  - rocprofv3 on PATH -- skipped gracefully if absent
  - minimap2 built with profile preset for trace logs
  - ptrace capability required in containers (for VTune/rocprof)

Output:
  {output_dir}/analysis.json     Structured analysis (always saved)
  {output_dir}/vtune_*/           VTune raw data
  {output_dir}/rocprof_*/         rocprofv3 raw data
  {output_dir}/trace.log          Trace timing log
  stdout                          Text summary (if --format text)

Key behaviors:
  - Forces -t 1 and -a flags for reproducible single-threaded profiling
  - Gracefully falls back if VTune or rocprofv3 are not installed
  - Classifies GPU kernels by ROCTX marker context + name patterns
  - Aggregates call counts and durations per function
  - Filters out external library functions (spdlog, STL, system)

Usage:
    python analyze_stages.py [options] -- <minimap2 args>

Examples:
    # CPU-only analysis
    python analyze_stages.py -- -ax map-ont ref.fa reads.fa

    # With GPU chaining
    python analyze_stages.py -- -ax map-ont --gpu-chain --gpu-cfg gpu/gpu_config.json --gpu-align ref.fa reads.fa

    # Skip VTune (rocprof + trace only)
    python analyze_stages.py --no-vtune -- -ax map-ont --gpu-chain --gpu-cfg gpu/gpu_config.json --gpu-align ref.fa reads.fa

    # Reuse saved profiles, just regenerate analysis
    python analyze_stages.py --reuse-data -o /tmp/mm2_analysis
"""

import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from stage_defs import PIPELINE_STAGES, PROFILE_MACRO_TO_STAGE

# =============================================================================
# Constants
# =============================================================================

# Cache for discovered task-to-stage mapping
_task_to_stage_cache: Optional[Dict[str, str]] = None


def _discover_task_to_stage() -> Dict[str, str]:
    """Discover task-to-stage mapping by scanning source code for MM_PROFILE_* macros."""
    global _task_to_stage_cache
    if _task_to_stage_cache is not None:
        return _task_to_stage_cache
    
    # Find the minimap2 source directory
    script_dir = Path(__file__).parent
    workspace_root = script_dir.parent
    minimap2_dir = workspace_root / 'minimap2'
    
    # Pattern to match MM_PROFILE_SEED("name"), MM_PROFILE_CHAIN("name"), etc.
    pattern = re.compile(r'MM_PROFILE_(SEED|CHAIN|ALIGN|IO|GPU)\s*\(\s*"([^"]+)"\s*\)')
    
    mapping: Dict[str, str] = {}
    
    if minimap2_dir.exists():
        # Scan all .c and .cpp files
        for ext in ['*.c', '*.cpp']:
            for src_file in minimap2_dir.glob(f'**/{ext}'):
                try:
                    content = src_file.read_text(errors='ignore')
                    for match in pattern.finditer(content):
                        macro_type = match.group(1)  # SEED, CHAIN, ALIGN, IO, GPU
                        task_name = match.group(2)   # The task name in quotes
                        
                        macro_name = f'MM_PROFILE_{macro_type}'
                        stage = PROFILE_MACRO_TO_STAGE.get(macro_name, 'other')
                        mapping[task_name] = stage
                except Exception:
                    pass
    
    # Add worker_* tasks based on their suffix
    mapping['worker_seed'] = 'seeding'
    mapping['worker_chain'] = 'chaining'
    mapping['worker_align'] = 'alignment'
    
    # Add I/O tasks
    mapping['bseq_read'] = 'io'
    mapping['bseq_write'] = 'io'
    
    # Fallback if scanning failed
    if not mapping:
        mapping = {
            'worker_seed': 'seeding', 'worker_chain': 'chaining', 'worker_align': 'alignment',
            'bseq_read': 'io', 'bseq_write': 'io',
        }
    
    _task_to_stage_cache = mapping
    return mapping


def get_task_to_stage() -> Dict[str, str]:
    """Get the task-to-stage mapping (cached after first call)."""
    return _discover_task_to_stage()



# Minimum duration threshold (microseconds) - filter out noise below this
# Set to 0 to show all functions regardless of duration
MIN_DURATION_US = 0


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TraceEvent:
    """Trace log event for timing."""
    timestamp: datetime
    event_type: str  # 'enter' or 'exit'
    function: str
    file: str
    line: int
    thread_id: int = 0
    duration_us: float = 0.0  # Duration in microseconds (for exit events)


@dataclass
class FunctionTiming:
    """Function timing from trace logs."""
    name: str
    file: str
    line: int
    duration_us: float
    call_count: int = 1
    children: List['FunctionTiming'] = field(default_factory=list)


@dataclass
class CPUFunction:
    """CPU function identified by VTune."""
    name: str
    module: str
    cpu_time_ms: float
    cpu_time_pct: float
    source_file: str = ''
    source_line: int = 0


@dataclass
class GPUFunction:
    """GPU function/kernel identified by rocprofv3."""
    name: str
    category: str  # 'kernel', 'hip_api', 'marker', 'memory'
    duration_ns: int
    call_count: int = 1
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageInfo:
    """Pipeline stage information."""
    name: str
    # Timing from trace logs
    trace_functions: List[FunctionTiming] = field(default_factory=list)
    total_trace_time_us: float = 0.0
    # CPU functions from VTune
    vtune_functions: List[CPUFunction] = field(default_factory=list)
    # GPU functions from rocprofv3
    gpu_kernels: List[GPUFunction] = field(default_factory=list)
    gpu_api_calls: List[GPUFunction] = field(default_factory=list)
    gpu_markers: List[GPUFunction] = field(default_factory=list)
    total_gpu_time_us: float = 0.0


# =============================================================================
# Trace Log Parser (for timing)
# =============================================================================

class TraceLogParser:
    """Parses minimap2 trace logs for timing information.
    
    Supports trace formats (spdlog pattern: [timestamp] [trace] [thread_id] [file:line] msg):
    - Enter: [timestamp] [trace] [tid] [file.c:line] >> func_name
    - Exit:  [timestamp] [trace] [tid] [file.c:line] << func_name [duration]
    
    Duration formats: 165.306us, 4.661ms, 1.234s, 801ns
    Also accepts the old format without [tid] for backward compatibility.
    """
    
    @staticmethod
    def parse_duration_to_us(duration_str: str) -> float:
        """Parse duration string like '165.306us', '4.661ms', '1.234s', '801ns' to microseconds."""
        if not duration_str:
            return 0.0
        
        duration_str = duration_str.strip()
        
        # Try to extract number and unit
        match = re.match(r'([\d.]+)\s*(ns|us|µs|ms|s)', duration_str, re.IGNORECASE)
        if not match:
            return 0.0
        
        value = float(match.group(1))
        unit = match.group(2).lower()
        
        if unit == 'ns':
            return value / 1000.0
        elif unit in ('us', 'µs'):
            return value
        elif unit == 'ms':
            return value * 1000.0
        elif unit == 's':
            return value * 1_000_000.0
        else:
            return 0.0
    
    # Pattern for enter events: [timestamp] [trace] [tid] [file:line] >> func_name
    # The [tid] group is optional for backward compatibility with old logs.
    ENTER_PATTERN = re.compile(
        r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\] \[trace\] '
        r'(?:\[(\d+)\] )?'
        r'\[(\w+\.\w+):(\d+)\]\s*'
        r'>>\s*(\w+)'
    )
    
    # Pattern for exit events: [timestamp] [trace] [tid] [file:line] << func_name [duration]
    # Duration formats: [165.306us], [4.661ms], [1.234s], [801ns]
    EXIT_PATTERN = re.compile(
        r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\] \[trace\] '
        r'(?:\[(\d+)\] )?'
        r'\[(\w+\.\w+):(\d+)\]\s*'
        r'<<\s*'
        r'(?:(\w*)\s*)?'           # Optional function name (may be empty)
        r'\[([^\]]+)\]'            # Duration in brackets (required)
    )
    
    # Fallback pattern for exit without duration
    EXIT_PATTERN_NO_DURATION = re.compile(
        r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\] \[trace\] '
        r'(?:\[(\d+)\] )?'
        r'\[(\w+\.\w+):(\d+)\]\s*'
        r'<<'
    )
    
    def __init__(self):
        self.events: List[TraceEvent] = []
    
    def reset(self):
        """Reset parser state for reuse."""
        self.events = []
    
    
    def parse_output(self, output: str) -> List[FunctionTiming]:
        """Parse trace output and build function timing hierarchy."""
        self.reset()  # Clear any previous state
        lines = output.split('\n')
        
        # Collect events
        for line in lines:
            # Skip lines without trace level
            if '[trace]' not in line:
                continue
            
            # Try enter pattern first
            match = self.ENTER_PATTERN.search(line)
            if match:
                timestamp_str, tid_str, file_name, line_num, func_name = match.groups()
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
                self.events.append(TraceEvent(
                    timestamp=timestamp,
                    event_type='enter',
                    function=func_name,
                    file=file_name,
                    line=int(line_num),
                    thread_id=int(tid_str) if tid_str else 0,
                ))
                continue
            
            # Try exit pattern with duration
            match = self.EXIT_PATTERN.search(line)
            if match:
                timestamp_str, tid_str, file_name, line_num, func_name, duration_str = match.groups()
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
                duration_us = self.parse_duration_to_us(duration_str)
                self.events.append(TraceEvent(
                    timestamp=timestamp,
                    event_type='exit',
                    function=func_name or '',
                    file=file_name,
                    line=int(line_num),
                    thread_id=int(tid_str) if tid_str else 0,
                    duration_us=duration_us,
                ))
                continue
            
            # Try fallback pattern without duration (old format)
            match = self.EXIT_PATTERN_NO_DURATION.search(line)
            if match:
                timestamp_str, tid_str, file_name, line_num = match.groups()
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S.%f')
                self.events.append(TraceEvent(
                    timestamp=timestamp,
                    event_type='exit',
                    function='',
                    file=file_name,
                    line=int(line_num),
                    thread_id=int(tid_str) if tid_str else 0,
                    duration_us=0.0,
                ))
        
        # Match enter/exit per thread using separate call stacks.
        # Multi-threaded minimap2 runs interleave trace lines from different
        # threads; a single stack would mis-pair enters and exits.
        # Per-thread stacks keyed by thread_id (0 = unknown / old logs).
        thread_stacks: Dict[int, List[Tuple[TraceEvent, FunctionTiming]]] = {}
        root_calls: List[FunctionTiming] = []
        
        for event in self.events:
            tid = event.thread_id
            if tid not in thread_stacks:
                thread_stacks[tid] = []
            call_stack = thread_stacks[tid]

            if event.event_type == 'enter':
                # Create a placeholder FunctionTiming
                func_timing = FunctionTiming(
                    name=event.function,
                    file=event.file,
                    line=event.line,
                    duration_us=0  # Will be set on exit
                )
                
                # If we have a parent on this thread's stack, add as child
                if call_stack:
                    call_stack[-1][1].children.append(func_timing)
                else:
                    # This is a root-level function for this thread
                    root_calls.append(func_timing)
                
                # Push onto this thread's stack
                call_stack.append((event, func_timing))
                
            elif event.event_type == 'exit' and call_stack:
                # Pop from this thread's stack
                enter_event, func_timing = call_stack.pop()
                # Use the duration from the exit event (parsed from trace log)
                # This is more accurate than calculating from timestamps
                if event.duration_us > 0:
                    func_timing.duration_us = event.duration_us
                else:
                    # Fallback to timestamp-based calculation (less accurate)
                    func_timing.duration_us = (event.timestamp - enter_event.timestamp).total_seconds() * 1_000_000
        
        return root_calls


# =============================================================================
# VTune Parser (for CPU function identification)
# =============================================================================

@dataclass
class TaskTiming:
    """ITT task timing from VTune."""
    name: str
    stage: str
    duration_s: float
    call_count: int
    
    
class VTuneParser:
    """Parses Intel VTune results for CPU function identification."""
    
    def __init__(self, result_dir: str):
        self.result_dir = Path(result_dir)
        self.functions: List[CPUFunction] = []
        self.task_timing: List[TaskTiming] = []
    
    def _get_cpu_freq(self, cursor=None) -> float:
        """Get CPU frequency in Hz. Tries VTune metadata, /proc/cpuinfo, then defaults."""
        # Try VTune metadata if cursor provided
        if cursor:
            try:
                cursor.execute("SELECT * FROM dd_core_frequency LIMIT 1")
                row = cursor.fetchone()
                if row:
                    # VTune stores frequency in various formats
                    return float(row[0]) * 1e6 if row[0] < 1e6 else float(row[0])
            except Exception:
                pass
        
        # Try /proc/cpuinfo on Linux
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if 'cpu MHz' in line:
                        mhz = float(line.split(':')[1].strip())
                        return mhz * 1e6
        except Exception:
            pass
        
        # Default to 2.4 GHz
        return 2.4e9
        
    def parse_sqlite_tasks(self) -> List[TaskTiming]:
        """Parse task timing directly from VTune SQLite database (most accurate)."""
        try:
            import sqlite3
        except ImportError:
            return []
        
        # Find the SQLite database
        db_path = self.result_dir / 'sqlite-db' / 'dicer.db'
        if not db_path.exists():
            # Try to find it
            for db_file in self.result_dir.glob('**/*.db'):
                db_path = db_file
                break
        
        if not db_path.exists():
            return []
        
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Get task types (ITT task names)
            cursor.execute('SELECT rowid, name FROM dd_task_type')
            task_types = {row[0]: row[1] for row in cursor.fetchall()}
            
            if not task_types:
                conn.close()
                return []
            
            # Aggregate task_data by task type (attr field maps to task type)
            cursor.execute('''
                SELECT attr, 
                       COUNT(*) as count,
                       SUM(end_tsc - start_tsc) as total_tsc
                FROM task_data
                GROUP BY attr
            ''')
            
            # Try to get CPU frequency from VTune metadata, fallback to /proc/cpuinfo, then default
            CPU_FREQ = self._get_cpu_freq(cursor)
            
            tasks = []
            for row in cursor.fetchall():
                attr, count, total_tsc = row
                task_name = task_types.get(attr, f'unknown_{attr}')
                stage = get_task_to_stage().get(task_name, 'other')
                duration_s = total_tsc / CPU_FREQ if total_tsc else 0
                
                tasks.append(TaskTiming(
                    name=task_name,
                    stage=stage,
                    duration_s=duration_s,
                    call_count=count
                ))
            
            conn.close()
            self.task_timing = sorted(tasks, key=lambda t: -t.duration_s)
            return self.task_timing
            
        except Exception as e:
            print(f"Warning: Error parsing VTune SQLite database: {e}")
            return []
    
    def parse_functions_by_task_timestamp(self) -> Dict[str, List[CPUFunction]]:
        """Correlate CPU functions with ITT tasks using timestamp data from SQLite.
        
        Returns dict of task_name -> list of CPUFunction that executed within that task's time range.
        This eliminates the need for hardcoded pattern matching.
        """
        try:
            import sqlite3
        except ImportError:
            return {}
        
        db_path = self.result_dir / 'sqlite-db' / 'dicer.db'
        if not db_path.exists():
            for db_file in self.result_dir.glob('**/*.db'):
                db_path = db_file
                break
        
        if not db_path.exists():
            return {}
        
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Get task types (ITT task names)
            cursor.execute('SELECT rowid, name FROM dd_task_type')
            task_types = {row[0]: row[1] for row in cursor.fetchall()}
            
            if not task_types:
                conn.close()
                return {}
            
            # Get all task time ranges, sorted by start time
            cursor.execute('''
                SELECT tt.name, td.start_tsc, td.end_tsc
                FROM task_data td
                JOIN dd_task_type tt ON td.attr = tt.rowid
                ORDER BY td.start_tsc
            ''')
            task_intervals = [(start, end, name) for name, start, end in cursor.fetchall()]
            
            if not task_intervals:
                conn.close()
                return {}
            
            # Stage priority for tie-breaking (higher = more specific, wins)
            stage_priority = {'alignment': 3, 'chaining': 2, 'seeding': 1, 'io': 0, 'other': -1}
            task_stage_map = get_task_to_stage()
            
            # Build sorted arrays for binary-search overlap detection.
            # task_intervals is already sorted by start_tsc.
            import bisect
            task_starts = [t[0] for t in task_intervals]
            
            def find_tasks_for_band(band_start, band_end):
                """Find the best matching task for a band's time range.
                
                Uses binary search (O(log n + k) where k = overlapping tasks)
                instead of linear scan over all tasks.
                """
                best_task = None
                best_overlap = 0
                best_priority = -2
                
                # Find insertion point: first task whose start >= band_end
                # Any task starting at or after band_end can't overlap.
                right = bisect.bisect_left(task_starts, band_end)
                
                # Walk backwards from there checking tasks that could overlap
                # A task overlaps if task_start < band_end AND task_end > band_start
                for i in range(right - 1, -1, -1):
                    task_start, task_end, task_name = task_intervals[i]
                    
                    # Since tasks are sorted by start, once task_end <= band_start
                    # we could stop — but task durations vary, so we use a heuristic:
                    # stop if task_start is far below band_start (beyond any plausible task)
                    if task_end <= band_start:
                        # Tasks are sorted by start_tsc, but end times vary.
                        # Once we've gone past enough non-overlapping tasks, bail.
                        if task_start < band_start:
                            break
                        continue
                    
                    overlap_start = max(band_start, task_start)
                    overlap_end = min(band_end, task_end)
                    if overlap_start >= overlap_end:
                        continue
                    
                    overlap_duration = overlap_end - overlap_start
                    stage = task_stage_map.get(task_name, 'other')
                    priority = stage_priority.get(stage, -1)
                    
                    if priority > best_priority or (priority == best_priority and overlap_duration > best_overlap):
                        best_task = task_name
                        best_overlap = overlap_duration
                        best_priority = priority
                
                if best_task:
                    return [(best_task, 1.0)]
                return [('__unassigned__', 1.0)]
            
            # Get function info
            cursor.execute('''
                SELECT rowid, name FROM dd_function
            ''')
            func_names = {row[0]: row[1] for row in cursor.fetchall()}
            
            cursor.execute('''
                SELECT rowid, name FROM dd_source_file
            ''')
            source_files = {row[0]: row[1] for row in cursor.fetchall()}
            
            cursor.execute('''
                SELECT rowid, src_file FROM dd_function
            ''')
            func_source = {row[0]: source_files.get(row[1], '') for row in cursor.fetchall()}
            
            # Get bands with their timestamps and function data
            # Each band has a time range and links to cpu_data_agg_data via rowid
            cursor.execute('''
                SELECT 
                    b.start_tsc,
                    b.end_tsc,
                    d.attr as func_id,
                    d.duration
                FROM cpu_data_agg_band b
                JOIN cpu_data_agg_data d ON d.rowid BETWEEN b.data_start_rowid AND b.data_end_rowid
                WHERE d.duration > 0
            ''')
            
            # Aggregate: task -> func_name -> {duration_ns, source_file}
            task_func_data: Dict[str, Dict[str, Dict]] = {}
            
            for start_tsc, end_tsc, func_id, duration in cursor.fetchall():
                # Find all overlapping tasks and distribute duration proportionally
                task_overlaps = find_tasks_for_band(start_tsc, end_tsc)
                
                func_name = func_names.get(func_id, f'func_{func_id}')
                source_file = func_source.get(func_id, '')
                
                for task_name, fraction in task_overlaps:
                    allocated_duration = duration * fraction
                    
                    if task_name not in task_func_data:
                        task_func_data[task_name] = {}
                    
                    if func_name not in task_func_data[task_name]:
                        task_func_data[task_name][func_name] = {
                            'duration_ns': 0,
                            'source_file': source_file
                        }
                    
                    task_func_data[task_name][func_name]['duration_ns'] += allocated_duration
            
            conn.close()
            
            # Convert to CPUFunction objects
            result: Dict[str, List[CPUFunction]] = {}
            for task_name, funcs in task_func_data.items():
                result[task_name] = []
                for func_name, data in funcs.items():
                    result[task_name].append(CPUFunction(
                        name=func_name,
                        module='',
                        cpu_time_ms=data['duration_ns'] / 1_000_000,  # ns to ms
                        cpu_time_pct=0.0,
                        source_file=data['source_file'],
                        source_line=0
                    ))
                # Sort by CPU time descending
                result[task_name].sort(key=lambda f: -f.cpu_time_ms)
            
            return result
            
        except Exception as e:
            print(f"Warning: Error correlating functions with tasks: {e}")
            import traceback
            traceback.print_exc()
            return {}
        
    def generate_report(self) -> bool:
        """Generate CSV report from VTune results."""
        csv_path = self.result_dir / 'hotspots.csv'
        cmd = [
            'vtune', '-report', 'hotspots',
            '-result-dir', str(self.result_dir),
            '-format', 'csv',
            '-report-output', str(csv_path)
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            return result.returncode == 0 and csv_path.exists()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def parse_csv(self, csv_path: Path) -> List[CPUFunction]:
        """Parse VTune CSV report."""
        functions = []
        if not csv_path.exists():
            return functions
            
        with open(csv_path, 'r') as f:
            lines = f.readlines()
        
        # Find header
        header_idx = 0
        for i, line in enumerate(lines):
            if 'Function' in line or 'function' in line.lower():
                header_idx = i
                break
        
        if header_idx >= len(lines) - 1:
            return functions
        
        # Auto-detect delimiter (VTune uses tabs, not commas)
        header_line = lines[header_idx].strip()
        delimiter = '\t' if '\t' in header_line else ','
        header = header_line.split(delimiter)
        
        for line in lines[header_idx + 1:]:
            if not line.strip():
                continue
            values = line.strip().split(delimiter)
            if len(values) < len(header):
                continue
            row = dict(zip(header, values))
            
            try:
                name = row.get('Function', row.get('function', 'unknown'))
                module = row.get('Module', row.get('module', ''))
                
                cpu_time_str = row.get('CPU Time', row.get('Self CPU Time', '0'))
                cpu_time_ms = float(cpu_time_str.replace('s', '').replace('m', '').strip()) * 1000
                
                cpu_pct_str = row.get('CPU Time:Self', row.get('% CPU Time', '0'))
                cpu_pct = float(cpu_pct_str.replace('%', '').strip())
                
                functions.append(CPUFunction(
                    name=name,
                    module=module,
                    cpu_time_ms=cpu_time_ms,
                    cpu_time_pct=cpu_pct,
                    source_file=row.get('Source File', ''),
                    source_line=int(row.get('Source Line', 0) or 0)
                ))
            except (ValueError, KeyError):
                continue
                
        return functions
    
    
    def parse_all(self) -> List[CPUFunction]:
        """Parse all VTune results."""
        if self.generate_report():
            csv_path = self.result_dir / 'hotspots.csv'
            self.functions = self.parse_csv(csv_path)
        
        if not self.functions:
            for csv_file in self.result_dir.glob('**/*.csv'):
                self.functions.extend(self.parse_csv(csv_file))
        
        self.functions.sort(key=lambda f: -f.cpu_time_ms)
        return self.functions
    


# =============================================================================
# rocprofv3 Parser (for GPU function identification)
# =============================================================================

class RocprofParser:
    """Parses rocprofv3 results for GPU function identification."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.functions: List[GPUFunction] = []
        
    def parse_sqlite(self, db_path: Path) -> List[GPUFunction]:
        """Parse rocprofv3 SQLite database with SQL-level aggregation.

        Uses GROUP BY queries instead of row-by-row iteration to reduce
        9M+ individual rows to ~30 aggregated entries.  The downstream
        code (aggregate_gpu, _build_report_dict) already aggregates by
        name, so per-row objects are unnecessary.

        Skips the rocpd_event table (no start/end/name_id columns –
        produces zero usable data despite millions of rows).
        """
        functions = []
        
        try:
            import sqlite3
        except ImportError:
            return functions
            
        if not db_path.exists():
            return functions
            
        try:
            conn = sqlite3.connect(str(db_path))
            # Enable memory-mapped I/O for faster reads on large databases
            conn.execute(f'PRAGMA mmap_size={db_path.stat().st_size * 2}')
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            def find_table(patterns):
                for table in tables:
                    for pattern in patterns:
                        if pattern.lower() in table.lower():
                            return table
                return None
            
            # Get string table
            string_map = {}
            string_table = find_table(['rocpd_string', 'string'])
            if string_table:
                try:
                    cursor.execute(f'SELECT id, string FROM "{string_table}"')
                    for row in cursor.fetchall():
                        string_map[row[0]] = row[1]
                except (sqlite3.Error, KeyError, IndexError):
                    pass
            
            # Get kernel symbol info for name resolution and resource usage
            # kernel_symbol table has: id, kernel_name, display_name,
            #   sgpr_count, arch_vgpr_count, accum_vgpr_count,
            #   group_segment_size (LDS), private_segment_size (scratch)
            kernel_symbols = {}
            symbol_table = find_table(['rocpd_info_kernel_symbol', 'kernel_symbol'])
            if symbol_table:
                try:
                    cursor.execute(f'SELECT * FROM "{symbol_table}"')
                    cols = [desc[0] for desc in cursor.description]
                    for row in cursor.fetchall():
                        row_dict = dict(zip(cols, row))
                        kid = row_dict.get('id')
                        # Prefer display_name, fallback to kernel_name
                        name = row_dict.get('display_name') or row_dict.get('kernel_name', f"kernel_{kid}")
                        if kid:
                            kernel_symbols[kid] = {
                                'name': name,
                                'resources': {
                                    'sgpr_count': row_dict.get('sgpr_count', 0),
                                    'arch_vgpr_count': row_dict.get('arch_vgpr_count', 0),
                                    'accum_vgpr_count': row_dict.get('accum_vgpr_count', 0),
                                    'group_segment_size': row_dict.get('group_segment_size', 0),
                                    'private_segment_size': row_dict.get('private_segment_size', 0),
                                },
                            }
                except (sqlite3.Error, KeyError, TypeError):
                    pass
            
            # Parse kernel dispatches — aggregated by kernel_id
            # Also fetches workgroup/grid dimensions needed for occupancy
            kernel_table = find_table(['rocpd_kernel_dispatch', 'kernel_dispatch'])
            if kernel_table:
                try:
                    cursor.execute(f'''
                        SELECT kernel_id,
                               SUM("end" - start)  AS total_duration,
                               COUNT(*)             AS cnt,
                               MAX(workgroup_size_x) AS wg_x,
                               MAX(workgroup_size_y) AS wg_y,
                               MAX(workgroup_size_z) AS wg_z,
                               MAX(grid_size_x * grid_size_y * grid_size_z) AS max_grid,
                               MIN(grid_size_x * grid_size_y * grid_size_z) AS min_grid
                        FROM "{kernel_table}"
                        WHERE start IS NOT NULL AND "end" IS NOT NULL
                        GROUP BY kernel_id
                    ''')
                    for row in cursor.fetchall():
                        kernel_id, total_dur, cnt, wg_x, wg_y, wg_z, max_grid, min_grid = row
                        sym = kernel_symbols.get(kernel_id, {})
                        if isinstance(sym, dict):
                            name = sym.get('name', f"kernel_{kernel_id}")
                            resources = sym.get('resources', {})
                        else:
                            name = sym or f"kernel_{kernel_id}"
                            resources = {}
                        
                        functions.append(GPUFunction(
                            name=str(name),
                            category='kernel',
                            duration_ns=int(total_dur),
                            call_count=int(cnt),
                            extra={
                                'kernel_resources': resources,
                                'workgroup_size_x': int(wg_x or 64),
                                'workgroup_size_y': int(wg_y or 1),
                                'workgroup_size_z': int(wg_z or 1),
                                'max_grid': int(max_grid or 64),
                                'min_grid': int(min_grid or 64),
                            }
                        ))
                except (sqlite3.Error, KeyError, TypeError, ValueError):
                    pass
            
            # Parse regions (HIP API calls) — aggregated by name_id
            # Skips roctx entries (roctxThreadRangeA etc.) which are the
            # roctx push/pop API calls, not user markers.
            region_table = find_table(['rocpd_region', 'region'])
            if region_table:
                try:
                    cursor.execute(f'''
                        SELECT name_id,
                               SUM("end" - start) AS total_duration,
                               COUNT(*)            AS cnt
                        FROM "{region_table}"
                        WHERE start IS NOT NULL AND "end" IS NOT NULL
                        GROUP BY name_id
                    ''')
                    for name_id, total_dur, cnt in cursor.fetchall():
                        name = string_map.get(name_id, str(name_id)) if name_id else 'unknown'
                        
                        # Skip internal roctx API calls
                        if name.startswith('roctx'):
                            continue
                        
                        if name.startswith('hip') or name.startswith('hsa') or name.startswith('__hip'):
                            category = 'hip_api'
                        else:
                            category = 'marker'
                        
                        functions.append(GPUFunction(
                            name=str(name),
                            category=category,
                            duration_ns=int(total_dur),
                            call_count=int(cnt),
                        ))
                except (sqlite3.Error, KeyError, TypeError, ValueError):
                    pass
            
            # NOTE: rocpd_event table is intentionally skipped.
            # It has no start/end/name_id columns (only id, guid,
            # category_id, stack_id, correlation_id, call_stack,
            # line_info, extdata) and produces zero GPUFunction objects
            # despite containing millions of rows.
            
            # Parse memory copies — aggregated by name_id (direction)
            memcpy_table = find_table(['rocpd_memory_copy', 'memory_copy'])
            if memcpy_table:
                try:
                    cursor.execute(f'''
                        SELECT name_id,
                               SUM("end" - start) AS total_duration,
                               COUNT(*)            AS cnt,
                               SUM(size)           AS total_bytes
                        FROM "{memcpy_table}"
                        WHERE start IS NOT NULL AND "end" IS NOT NULL
                        GROUP BY name_id
                    ''')
                    for name_id, total_dur, cnt, total_bytes in cursor.fetchall():
                        kind = string_map.get(name_id, 'memcpy') if name_id else 'memcpy'
                        
                        functions.append(GPUFunction(
                            name=f"memcpy_{kind}",
                            category='memory',
                            duration_ns=int(total_dur),
                            call_count=int(cnt),
                            extra={'copy_bytes': int(total_bytes) if total_bytes else 0}
                        ))
                except (sqlite3.Error, KeyError, TypeError, ValueError):
                    pass
            
            conn.close()
        except Exception as e:
            print(f"Warning: Error parsing rocprofv3 database: {e}")
            
        return functions
    
    def parse_all(self) -> List[GPUFunction]:
        """Parse all rocprofv3 output."""
        if not self.output_dir.exists():
            return []
        
        for db_file in self.output_dir.glob('**/*.db'):
            self.functions.extend(self.parse_sqlite(db_file))
        
        return self.functions


# =============================================================================
# Stage Analyzer
# =============================================================================

class StageAnalyzer:
    """Categorizes functions into pipeline stages.
    
    Uses get_task_to_stage() for mapping ITT/ROCTX task names to stages.
    The mapping is discovered dynamically from MM_PROFILE_* macros in source code.
    All other functions are classified by timestamp correlation with these tasks.
    """
    
    def __init__(self):
        self.stages: Dict[str, StageInfo] = {}
        for name in PIPELINE_STAGES:
            self.stages[name] = StageInfo(name=name)
    
    def classify_by_task(self, task_name: str) -> str:
        """Classify by ITT/ROCTX task name."""
        task_to_stage = get_task_to_stage()
        if task_name in task_to_stage:
            return task_to_stage[task_name]
        # Try partial match for the main worker tasks
        task_lower = task_name.lower()
        for task, stage in task_to_stage.items():
            if task.lower() in task_lower or task_lower in task.lower():
                return stage
        return 'other'
    
    def classify_gpu_marker(self, name: str) -> str:
        """Classify a ROCTX marker into a stage (used for GPU kernel context)."""
        return self.classify_by_task(name)
    
    def analyze(self, trace_funcs: List[FunctionTiming], 
                vtune_funcs: List[CPUFunction],
                gpu_funcs: List[GPUFunction],
                vtune_tasks: Optional[Dict[str, List[CPUFunction]]] = None):
        """Analyze and categorize all functions.
        
        Args:
            trace_funcs: Function timing from trace logs
            vtune_funcs: CPU functions from VTune (pattern-matched)
            gpu_funcs: GPU functions from rocprofv3
            vtune_tasks: Optional dict of ITT task -> functions (preferred for classification)
        """
        
        # Process trace timing - trace functions inherit stage from parent
        # Only top-level tasks (worker_seed, worker_chain, etc.) define stages
        # Child functions inherit their parent's stage
        #
        # total_trace_time_us only counts stage-entry-point functions to avoid
        # double-counting nested call chains (e.g. worker_align -> mm_map_align
        # -> align1 are all "alignment", but only worker_align's time should be
        # counted toward the stage total because children are already included
        # in their parent's duration).
        def process_trace(funcs: List[FunctionTiming], parent_stage: Optional[str] = None):
            for func in funcs:
                # Try to classify by task name first
                stage = self.classify_by_task(func.name)
                
                # If not a known task and we have a parent stage, inherit it
                if stage == 'other' and parent_stage is not None:
                    stage = parent_stage
                
                self.stages[stage].trace_functions.append(func)

                # Only accumulate time for stage entry points: functions that
                # either are at the top level (parent_stage is None) or that
                # define a NEW stage different from their parent's stage.
                is_stage_entry = (parent_stage is None or stage != parent_stage)
                if is_stage_entry:
                    self.stages[stage].total_trace_time_us += func.duration_us
                
                # Children inherit this function's stage
                if func.children:
                    process_trace(func.children, stage)
        
        process_trace(trace_funcs)
        
        # Process VTune CPU functions
        # vtune_tasks: dict of ITT task name -> list of functions that executed within that task
        # Functions are classified by:
        # 1. Function name/source patterns (most reliable for minimap2 functions)
        # 2. Task timestamp correlation (fallback)
        if vtune_tasks:
            for task_name, funcs in vtune_tasks.items():
                task_stage = 'other' if task_name.startswith('__') else self.classify_by_task(task_name)
                
                for func in funcs:
                    # First try to classify by function name patterns (most reliable)
                    func_stage = get_stage_from_function_name(func.name)
                    
                    # If no match, try source file path
                    if not func_stage:
                        func_stage = get_stage_from_source_path(func.source_file)
                    
                    # If still no match, use task-based classification
                    if not func_stage:
                        func_stage = task_stage
                    
                    self.stages[func_stage].vtune_functions.append(func)
        elif vtune_funcs:
            # No timestamp correlation available - classify by function name/source
            for func in vtune_funcs:
                func_stage = get_stage_from_function_name(func.name)
                if not func_stage:
                    func_stage = get_stage_from_source_path(func.source_file)
                if not func_stage:
                    func_stage = 'other'
                self.stages[func_stage].vtune_functions.append(func)
        
        def classify_kernel_by_name(kernel_name: str) -> str:
            """Classify GPU kernel by name pattern (fallback when ROCTX context unavailable).
            
            Kernel names indicate their purpose:
            - score_generation_*, range_selection_* -> chaining (GPU chain algorithm)
            - ksw_extd2_gpu, ksw_extz2_gpu -> alignment (GPU DP alignment)
            - __amd_rocclr_* -> internal (memory ops, not user kernels)
            """
            name_lower = kernel_name.lower()
            
            # Chaining kernels
            if any(p in name_lower for p in ['score_generation', 'range_selection', 'plchain']):
                return 'chaining'
            
            # Alignment kernels
            if any(p in name_lower for p in ['ksw_ext', 'ksw2', 'alignment', 'backtrack', 'cigar']):
                return 'alignment'
            
            # Internal ROCm kernels - don't classify
            if name_lower.startswith('__amd_rocclr'):
                return 'gpu_other'
            
            return 'gpu_other'
        
        # Process GPU functions
        # With aggregated data (SQL GROUP BY), timestamp-based marker context
        # is unavailable. Classify kernels by name pattern; put HIP API calls
        # and markers under gpu_other since per-call context is lost.
        #
        # If per-row data is available (non-aggregated), the sort-by-start +
        # ROCTX marker context tracking would work, but we don't rely on it.
        for func in gpu_funcs:
            if func.category == 'marker':
                stage = self.classify_gpu_marker(func.name)
                if stage == 'other':
                    stage = 'gpu_other'
                self.stages[stage].gpu_markers.append(func)
            elif func.category == 'kernel':
                stage = classify_kernel_by_name(func.name)
                self.stages[stage].gpu_kernels.append(func)
                self.stages[stage].total_gpu_time_us += func.duration_ns / 1000
            else:
                # HIP API calls and memory entries — no per-call context
                self.stages['gpu_other'].gpu_api_calls.append(func)


# =============================================================================
# Execution Functions
# =============================================================================

# =============================================================================
# GPU Memory Sampler (rocm-smi)
# =============================================================================

class GpuMemorySampler:
    """Polls rocm-smi and /proc in a background thread to capture memory usage.

    GPU VRAM: Uses ``rocm-smi --showpids --json`` to track per-process VRAM.
    Falls back to system-wide ``--showmeminfo vram`` if per-PID data is
    unavailable (e.g. no GPU context yet).

    CPU RSS: Scans ``/proc/<pid>/status`` for VmRSS of processes whose
    name contains *process_name*.

    Records baseline (before run), peak, and final usage for both.
    All samples include timestamps for timeline graph generation.
    Sampling interval is configurable (default 0.2s).
    """

    def __init__(self, process_name: str = 'minimap2', interval: float = 0.2):
        self.process_name = process_name.lower()
        self.interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()  # protects self.samples
        self.samples: List[Dict[str, Any]] = []
        self.baseline: Optional[Dict[str, Any]] = None
        self._vram_total: int = 0  # cached from system query
        self._start_time: float = 0.0
        self._cached_pids: List[str] = []  # cached matching PIDs
        self._pid_scan_counter: int = 0     # rescan /proc every N samples
        self._summary_cache: Optional[Dict[str, Any]] = None
        self._page_size: int = os.sysconf('SC_PAGE_SIZE')  # typically 4096

    # ----- low-level queries -------------------------------------------------

    def _query_system_vram_total(self) -> int:
        """One-shot query for total VRAM capacity (system-wide, constant)."""
        try:
            result = subprocess.run(
                ['rocm-smi', '--showmeminfo', 'vram', '--json'],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                return 0
            data = json.loads(result.stdout)
            for card in data.values():
                if isinstance(card, dict):
                    return int(card.get('VRAM Total Memory (B)', 0))
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError,
                json.JSONDecodeError, ValueError, KeyError):
            pass
        return 0

    def _query_pid_vram(self) -> Optional[Dict[str, int]]:
        """Query per-process VRAM via ``rocm-smi --showpids --json``.

        Returns the sum of VRAM used by all PIDs whose name contains
        *self.process_name*.  Returns None when no matching PID is found
        (the target process may not have an active GPU context yet).

        rocm-smi --showpids --json output format::

            {"system": {"PID<n>": "<name>, <gpu>, <vram_bytes>, <sdma>, <cu>"}}
        """
        try:
            result = subprocess.run(
                ['rocm-smi', '--showpids', '--json'],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                return None
            data = json.loads(result.stdout)
            system = data.get('system', {})
            total_vram = 0
            found = False
            for key, value in system.items():
                if not isinstance(value, str):
                    continue
                parts = [p.strip() for p in value.split(',')]
                if len(parts) < 3:
                    continue
                proc_name = parts[0].lower()
                if self.process_name not in proc_name:
                    continue
                try:
                    total_vram += int(parts[2])
                    found = True
                except (ValueError, IndexError):
                    continue
            if found:
                return {'vram_used_bytes': total_vram}
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError,
                json.JSONDecodeError, ValueError, KeyError):
            pass
        return None

    def _find_matching_pids(self) -> List[str]:
        """Scan /proc for PIDs whose comm matches *self.process_name*."""
        pids = []
        try:
            for entry in os.listdir('/proc'):
                if not entry.isdigit():
                    continue
                try:
                    with open(f'/proc/{entry}/comm') as f:
                        comm = f.read().strip().lower()
                    if self.process_name in comm:
                        pids.append(entry)
                except (OSError, ValueError):
                    continue
        except OSError:
            pass
        return pids

    def _query_pid_rss(self) -> Optional[int]:
        """Query RSS (bytes) for cached PIDs, re-scanning /proc periodically."""
        # Re-scan /proc every 25 samples (~5s at 0.2s interval) or if cache empty
        self._pid_scan_counter += 1
        if not self._cached_pids or self._pid_scan_counter >= 25:
            self._cached_pids = self._find_matching_pids()
            self._pid_scan_counter = 0

        total_rss = 0
        found = False
        stale_pids = []
        for pid in self._cached_pids:
            try:
                # statm is faster than parsing status: fields are in pages
                # Format: size resident shared text lib data dt
                with open(f'/proc/{pid}/statm') as f:
                    fields = f.read().split()
                resident_pages = int(fields[1])
                total_rss += resident_pages * self._page_size
                found = True
            except (OSError, ValueError, IndexError):
                stale_pids.append(pid)

        # Remove stale PIDs (process exited)
        for pid in stale_pids:
            self._cached_pids.remove(pid)

        return total_rss if found else None

    def _query(self) -> Optional[Dict[str, Any]]:
        """Best-effort VRAM query: per-process first, system-wide fallback."""
        pid_data = self._query_pid_vram()
        if pid_data is not None:
            pid_data['vram_total_bytes'] = self._vram_total
            pid_data['source'] = 'per_process'
            return pid_data
        # Fallback to system-wide (useful for baseline before GPU context exists)
        try:
            result = subprocess.run(
                ['rocm-smi', '--showmemuse', '--showmeminfo', 'vram', '--json'],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                return None
            data = json.loads(result.stdout)
            for card in data.values():
                if isinstance(card, dict):
                    return {
                        'vram_total_bytes': int(card.get('VRAM Total Memory (B)', 0)),
                        'vram_used_bytes': int(card.get('VRAM Total Used Memory (B)', 0)),
                        'source': 'system_wide',
                    }
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError,
                json.JSONDecodeError, ValueError, KeyError):
            pass
        return None

    # ----- sampling ----------------------------------------------------------

    def _sample_loop(self):
        while not self._stop.is_set():
            t = time.time() - self._start_time
            gpu_sample = self._query()
            cpu_rss = self._query_pid_rss()

            entry: Dict[str, Any] = {'time_s': round(t, 3)}
            has_data = False
            if gpu_sample:
                entry['vram_used_bytes'] = gpu_sample['vram_used_bytes']
                entry['source'] = gpu_sample.get('source', 'unknown')
                has_data = True
            if cpu_rss is not None:
                entry['cpu_rss_bytes'] = cpu_rss
                has_data = True
            if has_data:
                with self._lock:
                    self.samples.append(entry)
            self._stop.wait(self.interval)

    def snapshot_baseline(self):
        """Take a single reading before the workload starts."""
        self._vram_total = self._query_system_vram_total()
        gpu = self._query()
        cpu_rss = self._query_pid_rss()
        self.baseline = {
            'vram_used_bytes': gpu['vram_used_bytes'] if gpu else 0,
            'cpu_rss_bytes': cpu_rss if cpu_rss is not None else 0,
            'source': gpu.get('source', 'system_wide') if gpu else 'system_wide',
        }

    def start(self):
        """Begin background sampling."""
        self._stop.clear()
        with self._lock:
            self.samples = []
        self._summary_cache = None
        self._cached_pids = []       # force rescan for newly-spawned process
        self._pid_scan_counter = 0
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, Any]:
        """Stop sampling and return summary dict."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        return self.summarize()

    def summarize(self) -> Dict[str, Any]:
        """Return baseline, peak, delta, and sample count for GPU and CPU.

        Result is cached after the first call (safe to call from generate_graph).
        """
        if self._summary_cache is not None:
            return self._summary_cache

        with self._lock:
            samples = list(self.samples)  # snapshot under lock

        if not samples:
            return {}
        vram_total = self._vram_total or 0

        # GPU VRAM stats
        vram_values = [s['vram_used_bytes'] for s in samples
                       if 'vram_used_bytes' in s]
        per_proc = [s['vram_used_bytes'] for s in samples
                    if s.get('source') == 'per_process']
        baseline_vram = self.baseline.get('vram_used_bytes', 0) if self.baseline else 0

        gpu_peak = max(per_proc) if per_proc else (max(vram_values) if vram_values else 0)
        gpu_final = vram_values[-1] if vram_values else 0

        # CPU RSS stats
        rss_values = [s['cpu_rss_bytes'] for s in samples
                      if 'cpu_rss_bytes' in s]
        baseline_rss = self.baseline.get('cpu_rss_bytes', 0) if self.baseline else 0
        cpu_peak = max(rss_values) if rss_values else 0
        cpu_final = rss_values[-1] if rss_values else 0

        # Timeseries for graph (keep all points — typically a few hundred)
        timeseries = []
        for s in samples:
            point: Dict[str, Any] = {'t': s['time_s']}
            if 'vram_used_bytes' in s:
                point['gpu'] = s['vram_used_bytes']
            if 'cpu_rss_bytes' in s:
                point['cpu'] = s['cpu_rss_bytes']
            timeseries.append(point)

        summary = {
            'vram_total_bytes': vram_total,
            'vram_baseline_bytes': baseline_vram,
            'vram_peak_bytes': gpu_peak,
            'vram_final_bytes': gpu_final,
            'vram_delta_bytes': gpu_peak - baseline_vram,
            'cpu_rss_baseline_bytes': baseline_rss,
            'cpu_rss_peak_bytes': cpu_peak,
            'cpu_rss_final_bytes': cpu_final,
            'cpu_rss_delta_bytes': cpu_peak - baseline_rss,
            'sample_count': len(samples),
            'per_process_samples': len(per_proc),
            'cpu_rss_samples': len(rss_values),
            'tracking': 'per_process' if per_proc else 'system_wide',
            'timeseries': timeseries,
        }
        self._summary_cache = summary
        return summary

    def generate_graph(self, output_path: str):
        """Generate a memory timeline PNG using matplotlib."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("  Note: matplotlib not installed, skipping memory graph")
            return

        summary = self.summarize()  # uses cache, no recomputation
        ts = summary.get('timeseries', [])
        if len(ts) < 2:
            return

        gpu_t = [p['t'] for p in ts if 'gpu' in p]
        gpu_v = [p['gpu'] / (1 << 20) for p in ts if 'gpu' in p]
        cpu_t = [p['t'] for p in ts if 'cpu' in p]
        cpu_v = [p['cpu'] / (1 << 20) for p in ts if 'cpu' in p]

        fig, ax = plt.subplots(figsize=(12, 5))
        if gpu_t:
            ax.plot(gpu_t, gpu_v, label='GPU VRAM', color='#e74c3c', linewidth=1.5)
        if cpu_t:
            ax.plot(cpu_t, cpu_v, label='CPU RSS', color='#3498db', linewidth=1.5)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Memory (MiB)')
        ax.set_title('Memory Utilization Timeline')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Annotate peaks
        if gpu_v:
            peak_idx = gpu_v.index(max(gpu_v))
            ax.annotate(f'{gpu_v[peak_idx]:.0f} MiB',
                        xy=(gpu_t[peak_idx], gpu_v[peak_idx]),
                        fontsize=8, color='#e74c3c')
        if cpu_v:
            peak_idx = cpu_v.index(max(cpu_v))
            ax.annotate(f'{cpu_v[peak_idx]:.0f} MiB',
                        xy=(cpu_t[peak_idx], cpu_v[peak_idx]),
                        fontsize=8, color='#3498db')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"  Memory graph: {output_path}")


# =============================================================================
# GPU Architecture Limits (for occupancy calculation)
# =============================================================================

# Keyed by gfx target prefix.  _detect_gpu_arch() picks the best match.
GPU_ARCH_LIMITS = {
    'gfx94': {  # CDNA3 — MI300X / MI300A
        'max_waves_per_cu': 32,   # wave64 mode
        'vgpr_per_simd': 512,
        'simds_per_cu': 4,
        'lds_per_cu': 65536,      # 64 KiB
        'vgpr_granularity': 8,    # wave64 VGPR alloc granularity
    },
    'gfx90': {  # CDNA2 — MI210 / MI250X
        'max_waves_per_cu': 32,
        'vgpr_per_simd': 512,
        'simds_per_cu': 4,
        'lds_per_cu': 65536,
        'vgpr_granularity': 8,
    },
    'gfx103': {  # RDNA2 — RX 6000 series
        'max_waves_per_cu': 32,
        'vgpr_per_simd': 1024,
        'simds_per_cu': 2,
        'lds_per_cu': 65536,
        'vgpr_granularity': 8,
    },
}

_cached_arch_limits: Optional[Dict[str, Any]] = None


def _detect_gpu_arch() -> Dict[str, Any]:
    """Detect GPU arch via ``rocminfo`` and return matching limits (cached)."""
    global _cached_arch_limits
    if _cached_arch_limits is not None:
        return _cached_arch_limits
    try:
        result = subprocess.run(['rocminfo'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                low = line.lower()
                if 'gfx' in low:
                    for prefix in sorted(GPU_ARCH_LIMITS, key=len, reverse=True):
                        if prefix in low:
                            _cached_arch_limits = GPU_ARCH_LIMITS[prefix]
                            return _cached_arch_limits
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    _cached_arch_limits = GPU_ARCH_LIMITS['gfx94']  # default CDNA3
    return _cached_arch_limits


# =============================================================================
# Hardware Counter Collection (rocprofv3 --pmc)
# =============================================================================

# Counter sets — each list is one collection pass (app re-runs per pass).
# Keep passes small; SQ counters share hardware slots on gfx9.
PERF_COUNTER_PASSES = [
    # Pass 1: instruction mix & occupancy
    ['SQ_WAVES', 'SQ_INSTS_VALU', 'SQ_INSTS_SALU', 'SQ_INSTS_LDS',
     'SQ_INSTS_VMEM', 'GRBM_GUI_ACTIVE'],
    # Pass 2: memory bandwidth & LDS bank conflicts
    ['FETCH_SIZE', 'WRITE_SIZE', 'SQ_LDS_BANK_CONFLICT'],
]


def run_rocprofv3_counters(minimap2_path: str, mm_args: List[str],
                          output_dir: str,
                          log_dir: Optional[str] = None) -> bool:
    """Run minimap2 under rocprofv3 with hardware counter collection.

    Creates a multi-pass counter input file and runs rocprofv3.
    The application is re-run once per pass to collect different counter sets.
    """
    os.makedirs(output_dir, exist_ok=True)

    input_file = os.path.join(output_dir, 'counters.txt')
    with open(input_file, 'w') as f:
        for counter_set in PERF_COUNTER_PASSES:
            f.write(f"pmc: {' '.join(counter_set)}\n")

    cmd = [
        'rocprofv3',
        '-i', input_file,
        '-d', output_dir,
        '-f', 'csv',
        '--', minimap2_path,
    ] + mm_args

    print(f"[counters] Running: rocprofv3 -i counters.txt "
          f"({len(PERF_COUNTER_PASSES)} passes)...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Save counter run log
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, 'rocprof_counters_run.log')
            with open(log_path, 'w') as f:
                f.write(f"=== COMMAND ===\n{' '.join(cmd)}\n\n")
                f.write(f"=== RETURN CODE ===\n{result.returncode}\n\n")
                if result.stdout:
                    f.write(f"=== STDOUT ===\n{result.stdout}\n\n")
                if result.stderr:
                    f.write(f"=== STDERR ===\n{result.stderr}\n")
            print(f"      Saved run log: {log_path}")

        csv_files = list(Path(output_dir).glob('**/*.csv'))
        if csv_files:
            return True
        if result.returncode != 0 and result.stderr:
            print(f"[counters] Warning: {result.stderr[:200]}")
        return False
    except FileNotFoundError:
        print("[counters] Warning: rocprofv3 not found")
        return False


def parse_counter_results(output_dir: str) -> Dict[str, Dict]:
    """Parse rocprofv3 counter-collection CSV results.

    Returns dict of kernel_name -> {
        'dispatch_count': int,
        'resources': {'Arch_VGPR': int, 'SGPR': int, ...},
        'counters': {'SQ_WAVES': float, ...},
        'derived': {'valu_pct': float, ...},
    }
    """
    META_COLS = {
        'Dispatch_ID', 'GPU_ID', 'Queue_ID', 'Process_ID', 'Thread_ID',
        'Grid_Size', 'Workgroup_Size', 'LDS_Per_Workgroup',
        'Scratch_Per_Workitem', 'Arch_VGPR', 'Accum_VGPR', 'SGPR',
        'Wave_Size', 'Kernel_Name',
    }

    result: Dict[str, Dict] = {}
    output_path = Path(output_dir)
    csv_files = sorted(output_path.glob('**/*.csv'))

    for csv_file in csv_files:
        try:
            with open(csv_file, newline='') as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    kernel_name = row.get('Kernel_Name', '')
                    if not kernel_name:
                        continue

                    if kernel_name not in result:
                        result[kernel_name] = {
                            'dispatch_count': 0,
                            'resources': {},
                            'counters': {},
                        }

                    entry = result[kernel_name]
                    entry['dispatch_count'] += 1

                    # Resource info — keep values from first dispatch
                    if not entry['resources']:
                        for key in ('Arch_VGPR', 'Accum_VGPR', 'SGPR',
                                    'LDS_Per_Workgroup', 'Scratch_Per_Workitem',
                                    'Wave_Size'):
                            if key in row:
                                try:
                                    entry['resources'][key] = int(row[key])
                                except (ValueError, TypeError):
                                    pass
                        for key in ('Workgroup_Size', 'Grid_Size'):
                            if key in row:
                                try:
                                    entry['resources'][key] = [
                                        int(x) for x in
                                        row[key].strip('()').split(',')
                                    ]
                                except (ValueError, TypeError):
                                    pass

                    # Accumulate counter values across dispatches
                    for key, val in row.items():
                        if key not in META_COLS and val:
                            try:
                                entry['counters'][key] = (
                                    entry['counters'].get(key, 0) + float(val)
                                )
                            except (ValueError, TypeError):
                                pass
        except Exception as exc:
            print(f"Warning: Error parsing counter CSV {csv_file}: {exc}")

    # Compute derived metrics
    for entry in result.values():
        entry['derived'] = _compute_derived_metrics(
            entry['counters'], entry.get('resources'))

    return result


def _compute_derived_metrics(counters: Dict[str, float],
                             resources: Optional[Dict] = None) -> Dict[str, Any]:
    """Compute derived metrics from raw hardware counters and resource usage."""
    derived: Dict[str, Any] = {}

    total_insts = sum(
        counters.get(k, 0)
        for k in ('SQ_INSTS_VALU', 'SQ_INSTS_SALU',
                  'SQ_INSTS_LDS', 'SQ_INSTS_VMEM')
    )

    if total_insts > 0:
        derived['valu_pct'] = round(
            counters.get('SQ_INSTS_VALU', 0) / total_insts * 100, 1)
        derived['salu_pct'] = round(
            counters.get('SQ_INSTS_SALU', 0) / total_insts * 100, 1)
        derived['vmem_pct'] = round(
            counters.get('SQ_INSTS_VMEM', 0) / total_insts * 100, 1)
        derived['lds_pct'] = round(
            counters.get('SQ_INSTS_LDS', 0) / total_insts * 100, 1)

    waves = counters.get('SQ_WAVES', 0)
    if waves > 0:
        derived['insts_per_wave'] = round(total_insts / waves, 1)
        derived['valu_per_wave'] = round(
            counters.get('SQ_INSTS_VALU', 0) / waves, 1)

    lds_insts = counters.get('SQ_INSTS_LDS', 0)
    if lds_insts > 0:
        derived['lds_conflict_rate'] = round(
            counters.get('SQ_LDS_BANK_CONFLICT', 0) / lds_insts * 100, 2)

    fetch = counters.get('FETCH_SIZE', 0)
    write = counters.get('WRITE_SIZE', 0)
    if fetch > 0 or write > 0:
        derived['fetch_size'] = int(fetch)
        derived['write_size'] = int(write)

    # Heuristic classification
    valu_pct = derived.get('valu_pct', 0)
    vmem_pct = derived.get('vmem_pct', 0)
    if valu_pct >= 60:
        derived['classification'] = 'compute-bound'
    elif vmem_pct >= 40:
        derived['classification'] = 'memory-bound'
    else:
        derived['classification'] = 'mixed'

    # Theoretical occupancy (VGPR- and LDS-limited)
    if resources:
        arch = _detect_gpu_arch()
        arch_vgpr = resources.get('Arch_VGPR', 0)
        lds_per_wg = resources.get('LDS_Per_Workgroup', 0)
        wave_size = resources.get('Wave_Size', 64)
        wg_dims = resources.get('Workgroup_Size', [64])
        wg_size = 1
        for d in (wg_dims if isinstance(wg_dims, list) else [wg_dims]):
            wg_size *= d

        max_waves = arch['max_waves_per_cu']
        gran = arch['vgpr_granularity']

        # VGPR-limited waves per CU
        if arch_vgpr > 0:
            vgpr_alloc = math.ceil(arch_vgpr / gran) * gran
            waves_per_simd = arch['vgpr_per_simd'] // vgpr_alloc
            waves_vgpr = waves_per_simd * arch['simds_per_cu']
        else:
            waves_vgpr = max_waves

        # LDS-limited waves per CU
        if lds_per_wg > 0:
            max_wgs = arch['lds_per_cu'] // lds_per_wg
            waves_per_wg = max(1, math.ceil(wg_size / wave_size))
            waves_lds = max_wgs * waves_per_wg
        else:
            waves_lds = max_waves

        active_waves = min(waves_vgpr, waves_lds, max_waves)
        derived['occupancy_pct'] = round(active_waves / max_waves * 100, 1)
        derived['occupancy_waves'] = active_waves
        derived['occupancy_max_waves'] = max_waves
        # Identify the limiting factor
        if active_waves == waves_vgpr and waves_vgpr < waves_lds:
            derived['occupancy_limiter'] = 'VGPR'
        elif active_waves == waves_lds and waves_lds < waves_vgpr:
            derived['occupancy_limiter'] = 'LDS'
        else:
            derived['occupancy_limiter'] = 'max_waves'

    return derived


def compute_kernel_occupancy(gpu_funcs: List[GPUFunction],
                             num_cus: int = 304) -> List[Dict[str, Any]]:
    """Compute per-kernel occupancy and CU utilization from parsed GPU functions.

    Uses kernel resource data (VGPR/SGPR/LDS) attached to each GPUFunction.extra
    by parse_sqlite(), plus dispatch grid/workgroup sizes, to calculate:
    - Theoretical occupancy (VGPR- and LDS-limited waves per CU)
    - CU utilization (how many CUs are active per dispatch based on grid size)

    Args:
        gpu_funcs: List of GPUFunction objects with category='kernel'.
        num_cus: Number of Compute Units on the target GPU (default 304 for MI300X).

    Returns:
        List of per-kernel dicts sorted by total time descending, each containing
        resource usage, occupancy, and CU utilization data.
    """
    arch = _detect_gpu_arch()
    wave_size = 64
    max_waves_cu = arch['max_waves_per_cu']
    vgpr_per_simd = arch['vgpr_per_simd']
    simds_per_cu = arch['simds_per_cu']
    lds_per_cu = arch['lds_per_cu']
    gran = arch['vgpr_granularity']

    # Aggregate per kernel name
    kernel_agg: Dict[str, Dict[str, Any]] = {}
    for fn in gpu_funcs:
        if fn.category != 'kernel':
            continue
        name = fn.name
        # Skip internal/runtime kernels (ROCm runtime copy/fill helpers)
        if name.startswith('__amd_rocclr'):
            continue
        res = fn.extra.get('kernel_resources', {})

        wg_x = fn.extra.get('workgroup_size_x', 64)
        wg_y = fn.extra.get('workgroup_size_y', 1)
        wg_z = fn.extra.get('workgroup_size_z', 1)
        wg_size = wg_x * wg_y * wg_z

        # Use pre-aggregated grid sizes if available (from SQL GROUP BY),
        # else compute from per-dispatch grid dimensions
        max_grid = fn.extra.get('max_grid', 0)
        min_grid = fn.extra.get('min_grid', 0)
        if not max_grid:
            grid_x = fn.extra.get('grid_size_x', wg_size)
            grid_y = fn.extra.get('grid_size_y', 1)
            grid_z = fn.extra.get('grid_size_z', 1)
            max_grid = grid_x * grid_y * grid_z
            min_grid = max_grid

        if name not in kernel_agg:
            kernel_agg[name] = {
                'dispatch_count': 0,
                'total_ns': 0,
                'resources': res,
                'workgroup_size': wg_size,
                'max_grid': max_grid,
                'min_grid': min_grid,
            }
        agg = kernel_agg[name]
        agg['dispatch_count'] += fn.call_count
        agg['total_ns'] += fn.duration_ns
        agg['max_grid'] = max(agg['max_grid'], max_grid)
        agg['min_grid'] = min(agg['min_grid'], min_grid)

    results = []
    for name, agg in kernel_agg.items():
        res = agg['resources']
        arch_vgpr = res.get('arch_vgpr_count', 0)
        accum_vgpr = res.get('accum_vgpr_count', 0)
        total_vgpr = max(arch_vgpr, arch_vgpr + accum_vgpr)
        sgpr = res.get('sgpr_count', 0)
        lds_bytes = res.get('group_segment_size', 0)
        scratch_bytes = res.get('private_segment_size', 0)
        wg_size = agg['workgroup_size']
        waves_per_wg = max(1, math.ceil(wg_size / wave_size))

        # VGPR-limited occupancy
        if total_vgpr > 0:
            vgpr_alloc = math.ceil(total_vgpr / gran) * gran
            waves_per_simd_vgpr = vgpr_per_simd // vgpr_alloc
            waves_vgpr = min(waves_per_simd_vgpr * simds_per_cu, max_waves_cu)
        else:
            waves_vgpr = max_waves_cu

        # LDS-limited occupancy
        if lds_bytes > 0:
            max_wgs_lds = lds_per_cu // lds_bytes
            waves_lds = max_wgs_lds * waves_per_wg
        else:
            waves_lds = max_waves_cu

        eff_waves = min(waves_vgpr, waves_lds, max_waves_cu)
        occupancy_pct = round(eff_waves / max_waves_cu * 100, 1)

        if eff_waves == waves_vgpr and waves_vgpr < waves_lds:
            limiter = 'VGPR'
        elif eff_waves == waves_lds and waves_lds < waves_vgpr:
            limiter = 'LDS'
        else:
            limiter = 'max_waves'

        # CU utilization per dispatch (based on max grid size)
        total_waves = agg['max_grid'] // wave_size
        cu_coverage_pct = round(min(total_waves / num_cus * 100, 100), 1)

        results.append({
            'name': name,
            'dispatch_count': agg['dispatch_count'],
            'total_time_us': round(agg['total_ns'] / 1000, 1),
            'avg_time_us': round(agg['total_ns'] / agg['dispatch_count'] / 1000, 1),
            'workgroup_size': wg_size,
            'max_grid_size': agg['max_grid'],
            'min_grid_size': agg['min_grid'],
            'arch_vgpr': arch_vgpr,
            'accum_vgpr': accum_vgpr,
            'sgpr': sgpr,
            'lds_bytes': lds_bytes,
            'scratch_bytes': scratch_bytes,
            'occupancy_pct': occupancy_pct,
            'occupancy_waves_per_cu': eff_waves,
            'occupancy_max_waves_per_cu': max_waves_cu,
            'occupancy_limiter': limiter,
            'waves_per_dispatch': total_waves,
            'cu_coverage_pct': cu_coverage_pct,
            'num_cus': num_cus,
        })

    results.sort(key=lambda x: -x['total_time_us'])
    return results


def check_tool(name: str) -> bool:
    """Check if a tool is available."""
    try:
        # For vtune, just check if it exists (--version can be slow)
        if name == 'vtune':
            result = subprocess.run(['which', name], capture_output=True, timeout=5)
            return result.returncode == 0
        result = subprocess.run([name, '--version'], capture_output=True, timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def normalize_args_for_profiling(mm_args: List[str]) -> List[str]:
    """Normalize minimap2 arguments for profiling.
    
    Enforces:
    - Single-threaded mode (-t 1) for accurate profiling:
      - Avoids timing overlaps between threads
      - Provides consistent, reproducible results
      - Simplifies stage timing correlation
    - SAM output (-a) to trigger CIGAR alignment:
      - Without -a, alignment stage is skipped (no CIGAR computation)
      - GPU alignment kernels require -a to be triggered
    """
    args = list(mm_args)  # Copy to avoid modifying original
    
    # Remove any existing -t or --threads arguments
    i = 0
    while i < len(args):
        if args[i] in ('-t', '--threads'):
            # Remove -t and its value
            args.pop(i)
            if i < len(args) and not args[i].startswith('-'):
                args.pop(i)
        elif args[i].startswith('-t') and len(args[i]) > 2:
            # Handle -t4 format
            args.pop(i)
        else:
            i += 1
    
    # Check if -a or -c (CIGAR output) is already present
    has_cigar_output = any(arg in ('-a', '-c', '--sam', '--cs') for arg in args)
    
    # Build normalized args: -t 1 first, then -a if needed
    normalized = ['-t', '1']
    if not has_cigar_output:
        normalized.append('-a')  # SAM output to trigger CIGAR alignment
    
    return normalized + args


def run_minimap2_trace(minimap2_path: str, mm_args: List[str],
                      log_dir: Optional[str] = None) -> str:
    """Run minimap2 and capture trace output.
    
    Args:
        minimap2_path: Path to minimap2 executable
        mm_args: Arguments to pass to minimap2
        log_dir: Directory to save trace.log and run log
    
    Returns:
        stderr output containing trace logs
    """
    cmd = [minimap2_path] + mm_args
    print(f"[Trace] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        trace_log = os.path.join(log_dir, 'trace.log')
        with open(trace_log, 'w') as f:
            f.write(result.stderr)
        print(f"      Saved trace log: {trace_log}")
        # Save stdout (SAM output) log if non-empty
        if result.stdout:
            run_log = os.path.join(log_dir, 'trace_stdout.log')
            with open(run_log, 'w') as f:
                f.write(result.stdout)
    return result.stderr


def run_vtune(minimap2_path: str, mm_args: List[str], output_dir: str,
              log_dir: Optional[str] = None) -> bool:
    """Run minimap2 under VTune.
    
    Note: VTune requires ptrace access. In containers or restricted environments,
    you may need to set: sudo sysctl kernel.yama.ptrace_scope=0
    """
    # Clean up stale VTune result dir (VTune -result-dir fails if it exists)
    if os.path.exists(output_dir):
        print(f"[VTune] Cleaning stale result dir: {output_dir}")
        shutil.rmtree(output_dir)

    # Try software sampling (doesn't require hardware PMU access)
    cmd = [
        'vtune', '-collect', 'hotspots',
        '-knob', 'sampling-mode=sw',
        '-knob', 'enable-stack-collection=true',
        '-result-dir', output_dir,
        '--', minimap2_path
    ] + mm_args
    
    print(f"[VTune] Running: vtune -collect hotspots ...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Save VTune run log
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, 'vtune_run.log')
            with open(log_path, 'w') as f:
                f.write(f"=== COMMAND ===\n{' '.join(cmd)}\n\n")
                f.write(f"=== RETURN CODE ===\n{result.returncode}\n\n")
                if result.stdout:
                    f.write(f"=== STDOUT ===\n{result.stdout}\n\n")
                if result.stderr:
                    f.write(f"=== STDERR ===\n{result.stderr}\n")
            print(f"      Saved run log: {log_path}")

        if result.returncode == 0:
            return True
        
        # Check for ptrace error
        if 'ptrace' in result.stderr.lower():
            print("[VTune] Error: ptrace access denied. In containers, run:")
            print("        sudo sysctl kernel.yama.ptrace_scope=0")
            print("        Or run VTune on a host system without container restrictions.")
        elif result.stderr:
            print(f"[VTune] Error: {result.stderr[:200]}")
        
        # Clean up failed result directory
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        return False
        
    except FileNotFoundError:
        print("[VTune] Not found")
        return False


def run_rocprofv3(minimap2_path: str, mm_args: List[str], output_dir: str,
                  log_dir: Optional[str] = None) -> bool:
    """Run minimap2 under rocprofv3.
    
    Captures kernel dispatches, HIP API calls, memory operations, and ROCTX markers.
    Tries --memory-allocation-trace first; falls back without it for older rocprofv3.
    Returns True only if profiling succeeded and produced output files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    base_flags = [
        '--kernel-trace',       # GPU kernel dispatches
        '--hip-trace',          # HIP API calls  
        '--memory-copy-trace',  # Memory transfer operations
        '--marker-trace',       # ROCTX markers
        '--kernel-rename',      # Rename kernels with enclosing ROCTX region names
    ]
    optional_flags = ['--memory-allocation-trace']  # may not exist in older versions

    for _, extra in enumerate([optional_flags, []]):
        cmd = ['rocprofv3'] + base_flags + extra + ['-d', output_dir, '--', minimap2_path] + mm_args

        label = 'rocprofv3 --kernel-trace --hip-trace --marker-trace'
        if extra:
            label += ' --memory-allocation-trace'
        print(f"[rocprofv3] Running: {label} ...")

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)

            # Save rocprofv3 run log
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
                attempt = 'with_memalloc' if extra else 'base'
                log_path = os.path.join(log_dir, f'rocprof_run_{attempt}.log')
                with open(log_path, 'w') as f:
                    f.write(f"=== COMMAND ===\n{' '.join(cmd)}\n\n")
                    f.write(f"=== RETURN CODE ===\n{result.returncode}\n\n")
                    if result.stdout:
                        f.write(f"=== STDOUT ===\n{result.stdout}\n\n")
                    if result.stderr:
                        f.write(f"=== STDERR ===\n{result.stderr}\n")
                print(f"      Saved run log: {log_path}")

            db_files = list(Path(output_dir).glob('**/*.db'))
            if db_files:
                return True

            if result.returncode != 0 and extra:
                # First attempt with optional flags failed — retry without
                print("[rocprofv3] Retrying without --memory-allocation-trace ...")
                # Clean stale output before retry
                for f in Path(output_dir).iterdir():
                    if f.is_file():
                        f.unlink()
                continue

            if result.returncode != 0 and result.stderr:
                print(f"[rocprofv3] Warning: {result.stderr[:200]}")
            return False

        except FileNotFoundError:
            print("[rocprofv3] Warning: rocprofv3 not found")
            return False

    return False


# =============================================================================
# Report Generation
# =============================================================================

def format_duration(us: float) -> str:
    """Format duration in human-readable form."""
    if us < 1000:
        return f"{us:.2f} µs"
    elif us < 1_000_000:
        return f"{us/1000:.2f} ms"
    else:
        return f"{us/1_000_000:.2f} s"


def format_bytes(b: int) -> str:
    """Format byte count in human-readable form."""
    if b >= 1 << 30:
        return f"{b / (1 << 30):.2f} GiB"
    if b >= 1 << 20:
        return f"{b / (1 << 20):.2f} MiB"
    if b >= 1 << 10:
        return f"{b / (1 << 10):.2f} KiB"
    return f"{b} B"


def aggregate_trace(funcs: List[FunctionTiming], min_duration_us: Optional[float] = None) -> Dict[str, FunctionTiming]:
    """Aggregate trace functions by name, filtering out noise below threshold."""
    if min_duration_us is None:
        min_duration_us = MIN_DURATION_US
    agg: Dict[str, FunctionTiming] = {}
    for f in funcs:
        if f.duration_us < min_duration_us:
            continue
        if f.name in agg:
            agg[f.name].duration_us += f.duration_us
            agg[f.name].call_count += 1
        else:
            agg[f.name] = FunctionTiming(
                name=f.name, file=f.file, line=f.line,
                duration_us=f.duration_us, call_count=1
            )
    return agg


def is_named_function(name: str) -> bool:
    """Check if function has proper debug symbol (not system-generated like func@0x... or func_NNN)."""
    if not name:
        return False
    # Filter out unnamed functions from ROCm/system libraries
    if name.startswith('func@0x'):
        return False
    if name.startswith('func_') and name[5:].isdigit():
        return False
    if name.startswith('unknown_attr_'):
        return False
    return True


def get_stage_from_source_path(source_file: str) -> Optional[str]:
    """Determine stage from source file path (fallback when task correlation fails).
    
    Uses directory structure: minimap2/align/src/ -> alignment, minimap2/seed/src/ -> seeding, etc.
    Note: hit.c is in /chain/ but contains both chaining and alignment functions, so we don't use it.
    """
    if not source_file:
        return None
    
    source_lower = source_file.lower()
    
    # Check directory-based stage (alignment first - higher priority)
    if '/align/' in source_lower or 'align.c' in source_lower or 'esterr' in source_lower:
        return 'alignment'
    if '/seed/' in source_lower or 'sketch' in source_lower or 'index' in source_lower:
        return 'seeding'
    # Note: hit.c contains both chaining (hit_sort) and alignment (gen_regs, filter_regs, etc.)
    # So we only use lchain.c and pe.c for chaining detection
    if 'lchain' in source_lower or 'pe.c' in source_lower or 'chain_map' in source_lower:
        return 'chaining'
    if 'bseq' in source_lower:
        return 'io'
    
    # Check function name patterns in source file
    if 'ksw2' in source_lower:  # SSE alignment kernels
        return 'alignment'
    
    return None


def get_stage_from_function_name(func_name: str) -> Optional[str]:
    """Determine stage from function name patterns (fallback when task correlation fails)."""
    if not func_name:
        return None
    
    name_lower = func_name.lower()
    
    # Alignment patterns (check first - higher priority)
    # Includes post-chaining functions that prepare for alignment
    if any(p in name_lower for p in [
        'align', 'ksw', 'cigar', 'zdrop',
        'gen_regs', 'filter_regs', 'set_mapq', 'set_parent',  # post-chain alignment prep
        'mapq', 'est_err'
    ]):
        return 'alignment'
    
    # Chaining patterns
    if any(p in name_lower for p in ['chain', 'lchain', 'rmq', 'hit_sort']):
        return 'chaining'
    
    # Seeding patterns
    if any(p in name_lower for p in ['seed', 'sketch', 'minimizer', 'idx_gen', 'idx_get', 'idx_load']):
        return 'seeding'
    
    # I/O patterns
    if any(p in name_lower for p in ['bseq', 'kseq']):
        return 'io'
    
    return None


# Cache for discovered minimap2 source files
_minimap2_source_cache: Optional[Set[str]] = None


def _discover_minimap2_sources() -> Set[str]:
    """Discover minimap2 source files by scanning the minimap2 directory."""
    global _minimap2_source_cache
    if _minimap2_source_cache is not None:
        return _minimap2_source_cache
    
    sources = set()
    
    # Find the minimap2 source directory relative to this script
    script_dir = Path(__file__).parent
    workspace_root = script_dir.parent
    minimap2_dir = workspace_root / 'minimap2'
    
    if minimap2_dir.exists():
        # Scan for all .c, .cpp, .h files in minimap2/*/src/ and minimap2/src/
        for pattern in ['**/*.c', '**/*.cpp']:
            for src_file in minimap2_dir.glob(pattern):
                # Get just the filename
                sources.add(src_file.name)
        
        # Also add files from lib/ directory (ksw2, etc.)
        lib_dir = workspace_root / 'lib'
        if lib_dir.exists():
            for pattern in ['*.c', '*.cpp']:
                for src_file in lib_dir.glob(pattern):
                    sources.add(src_file.name)
    
    # Fallback: add common minimap2 source files if directory scan fails
    if not sources:
        sources = {
            'align.c', 'seed.c', 'chain.c', 'lchain.c', 'hit.c', 'sketch.c', 'index.c',
            'map.c', 'bseq.c', 'format.c', 'misc.c', 'kalloc.c', 'esterr.c', 'pe.c',
            'seed_map.c', 'chain_map.c', 'kthread.c', 'main.c',
            'mm_log.cpp', 'mm_profiler.cpp',
            'ksw2_ll_sse.c', 'ksw2_extd2_sse.c', 'ksw2_extz2_sse.c', 'ksw2_exts2_sse.c',
        }
    
    _minimap2_source_cache = sources
    return sources


# Known external library patterns to exclude
EXTERNAL_LIB_PATTERNS = [
    'spdlog', 'fmt/', 'std_', 'shared_ptr', 'unique_lock', 'condition_variable',
    'alloc_traits', 'stl_', 'vector.tcc', 'invoke.h', 'gthr-', 'emmintrin.h',
    'smmintrin.h', 'thread_pool', 'async_logger', 'pattern_formatter',
    'mpmc_blocking', 'log_msg_buffer', 'ittnotify', 'new_allocator',
    'atomic', 'mutex', 'functional', 'memory', 'chrono',
]


def is_minimap2_source(source_file: str) -> bool:
    """Check if a source file is from minimap2 codebase (not external libs like spdlog, STL, etc.)."""
    if not source_file:
        return False
    
    # Exclude known external libraries first (fast path)
    for pattern in EXTERNAL_LIB_PATTERNS:
        if pattern in source_file:
            return False
    
    # Extract just the filename from path
    filename = source_file.split('/')[-1].split('\\')[-1]
    
    # Check against discovered minimap2 source files
    minimap2_sources = _discover_minimap2_sources()
    if filename in minimap2_sources:
        return True
    
    # Also match by path pattern (minimap2/*/src/*.c or similar)
    source_lower = source_file.lower()
    if 'minimap2' in source_lower and '/src/' in source_lower:
        return True
    
    return False


def aggregate_vtune_unnamed(funcs: List[CPUFunction], min_time_ms: Optional[float] = None) -> Dict[str, Dict]:
    """Aggregate only unnamed/system-generated VTune CPU functions (for separate reporting)."""
    if min_time_ms is None:
        min_time_ms = MIN_DURATION_US / 1000
    agg: Dict[str, Dict] = {}
    for f in funcs:
        if f.cpu_time_ms < min_time_ms:
            continue
        # Only include unnamed/system-generated functions
        if is_named_function(f.name):
            continue
        if f.name in agg:
            agg[f.name]['cpu_time_ms'] += f.cpu_time_ms
            agg[f.name]['count'] += 1
        else:
            agg[f.name] = {
                'cpu_time_ms': f.cpu_time_ms,
                'count': 1,
                'source_file': f.source_file,
                'module': f.module
            }
    return agg


def is_loop_entry(name: str) -> bool:
    """Check if function name is a VTune loop entry like '[Loop at line X in func]' or '[Loop@0x...]'."""
    if not name:
        return False
    return name.startswith('[Loop')


def aggregate_vtune_minimap2(funcs: List[CPUFunction], min_time_ms: Optional[float] = None,
                              include_loops: bool = False) -> Dict[str, Dict]:
    """Aggregate only minimap2 source functions (from minimap2/*/src).
    
    Args:
        funcs: List of CPU functions
        min_time_ms: Minimum time threshold
        include_loops: If False, filter out VTune loop entries like '[Loop at line X in func]'
    """
    if min_time_ms is None:
        min_time_ms = MIN_DURATION_US / 1000
    agg: Dict[str, Dict] = {}
    for f in funcs:
        if f.cpu_time_ms < min_time_ms:
            continue
        # Only include named functions from minimap2 sources
        if not is_named_function(f.name):
            continue
        if not is_minimap2_source(f.source_file):
            continue
        # Filter out loop entries if requested
        if not include_loops and is_loop_entry(f.name):
            continue
        if f.name in agg:
            agg[f.name]['cpu_time_ms'] += f.cpu_time_ms
            agg[f.name]['count'] += 1
        else:
            agg[f.name] = {
                'cpu_time_ms': f.cpu_time_ms,
                'count': 1,
                'source_file': f.source_file,
                'module': f.module
            }
    return agg


def aggregate_vtune_external(funcs: List[CPUFunction], min_time_ms: Optional[float] = None) -> Dict[str, Dict]:
    """Aggregate named functions NOT from minimap2 sources (external libs like spdlog, STL, etc.)."""
    if min_time_ms is None:
        min_time_ms = MIN_DURATION_US / 1000
    agg: Dict[str, Dict] = {}
    for f in funcs:
        if f.cpu_time_ms < min_time_ms:
            continue
        # Only include named functions NOT from minimap2 sources
        if not is_named_function(f.name):
            continue
        if is_minimap2_source(f.source_file):
            continue
        if f.name in agg:
            agg[f.name]['cpu_time_ms'] += f.cpu_time_ms
            agg[f.name]['count'] += 1
        else:
            agg[f.name] = {
                'cpu_time_ms': f.cpu_time_ms,
                'count': 1,
                'source_file': f.source_file,
                'module': f.module
            }
    return agg


def aggregate_gpu(funcs: List[GPUFunction], min_duration_ns: Optional[float] = None) -> Dict[str, Dict]:
    """Aggregate GPU functions by name, filtering out noise."""
    if min_duration_ns is None:
        min_duration_ns = MIN_DURATION_US * 1000
    agg: Dict[str, Dict] = {}
    for f in funcs:
        if f.duration_ns < min_duration_ns:
            continue
        if f.name in agg:
            agg[f.name]['duration_ns'] += f.duration_ns
            agg[f.name]['count'] += f.call_count
            agg[f.name]['bytes_total'] += f.extra.get('copy_bytes', 0)
        else:
            agg[f.name] = {
                'duration_ns': f.duration_ns, 'count': f.call_count, 'category': f.category,
                'bytes_total': f.extra.get('copy_bytes', 0),
            }
    return agg


def _build_report_dict(analyzer: StageAnalyzer,
                       kernel_counters: Optional[Dict[str, Dict]] = None,
                       gpu_memory: Optional[Dict[str, Any]] = None,
                       extra_summary: Optional[Dict[str, Any]] = None,
                       kernel_occupancy: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Build the JSON report dict used by both --format json and analysis.json save."""
    total_trace = sum(s.total_trace_time_us for s in analyzer.stages.values())
    total_gpu = sum(s.total_gpu_time_us for s in analyzer.stages.values())

    summary: Dict[str, Any] = {
        'total_trace_time_us': total_trace,
        'total_gpu_time_us': total_gpu,
    }
    if gpu_memory:
        summary['gpu_memory'] = gpu_memory
    if extra_summary:
        summary.update(extra_summary)

    report: Dict[str, Any] = {
        'summary': summary,
        'stages': {},
        'kernel_counters': kernel_counters or {},
        'kernel_occupancy': kernel_occupancy or [],
    }

    for name, stage in analyzer.stages.items():
        if (not stage.trace_functions and not stage.vtune_functions
                and not stage.gpu_kernels and not stage.gpu_api_calls
                and not stage.gpu_markers):
            continue

        vtune_mm2 = aggregate_vtune_minimap2(stage.vtune_functions)
        vtune_ext = aggregate_vtune_external(stage.vtune_functions)
        vtune_unnamed = aggregate_vtune_unnamed(stage.vtune_functions)

        report['stages'][name] = {
            'trace_time_us': stage.total_trace_time_us,
            'gpu_time_us': stage.total_gpu_time_us,
            'trace_functions': [{
                'name': f.name, 'file': f.file, 'duration_us': f.duration_us, 'count': f.call_count
            } for f in sorted(aggregate_trace(stage.trace_functions).values(),
                              key=lambda x: -x.duration_us)],
            'vtune_minimap2_functions': [{
                'name': n, 'cpu_time_ms': d['cpu_time_ms'], 'count': d['count'],
                'source_file': d['source_file']
            } for n, d in sorted(vtune_mm2.items(), key=lambda x: -x[1]['cpu_time_ms'])],
            'vtune_minimap2_total_ms': sum(d['cpu_time_ms'] for d in vtune_mm2.values()),
            'vtune_external_functions': [{
                'name': n, 'cpu_time_ms': d['cpu_time_ms'], 'count': d['count'],
                'source_file': d['source_file']
            } for n, d in sorted(vtune_ext.items(), key=lambda x: -x[1]['cpu_time_ms'])[:20]],
            'vtune_external_total_ms': sum(d['cpu_time_ms'] for d in vtune_ext.values()),
            'vtune_external_count': len(vtune_ext),
            'vtune_unnamed_functions': [{
                'name': n, 'cpu_time_ms': d['cpu_time_ms'], 'count': d['count']
            } for n, d in sorted(vtune_unnamed.items(), key=lambda x: -x[1]['cpu_time_ms'])[:20]],
            'vtune_unnamed_total_ms': sum(d['cpu_time_ms'] for d in vtune_unnamed.values()),
            'vtune_unnamed_count': len(vtune_unnamed),
            'gpu_kernels': [{
                'name': n, 'duration_us': d['duration_ns']/1000, 'count': d['count']
            } for n, d in sorted(aggregate_gpu(stage.gpu_kernels).items(),
                                 key=lambda x: -x[1]['duration_ns'])],
            'gpu_api_calls': [{
                'name': n, 'duration_us': d['duration_ns']/1000, 'count': d['count'],
                'bytes_total': d['bytes_total'],
            } for n, d in sorted(aggregate_gpu(stage.gpu_api_calls).items(),
                                 key=lambda x: -x[1]['duration_ns'])],
            'gpu_markers': [{
                'name': n, 'duration_us': d['duration_ns']/1000, 'count': d['count']
            } for n, d in sorted(aggregate_gpu(stage.gpu_markers).items(),
                                 key=lambda x: -x[1]['duration_ns'])],
        }

    return report


def print_report(analyzer: StageAnalyzer, output_format: str = 'text',
                 kernel_counters: Optional[Dict[str, Dict]] = None,
                 gpu_memory: Optional[Dict[str, Any]] = None):
    """Print analysis report."""
    
    total_trace = sum(s.total_trace_time_us for s in analyzer.stages.values())
    total_gpu = sum(s.total_gpu_time_us for s in analyzer.stages.values())
    
    if output_format == 'json':
        report = _build_report_dict(analyzer, kernel_counters, gpu_memory)
        print(json.dumps(report, indent=2))
        return
    
    # Text format
    print("\n" + "=" * 80)
    print("MINIMAP2 PIPELINE STAGE ANALYSIS")
    print("=" * 80)
    print(f"\nTiming (trace logs): {format_duration(total_trace)}")
    print(f"GPU Time: {format_duration(total_gpu)}")
    
    for stage_name in PIPELINE_STAGES:
        stage = analyzer.stages.get(stage_name)
        if not stage:
            continue
        if (not stage.trace_functions and not stage.vtune_functions
                and not stage.gpu_kernels and not stage.gpu_api_calls
                and not stage.gpu_markers):
            continue
        
        print(f"\n{'─' * 80}")
        print(f"STAGE: {stage_name.upper()}")
        print(f"{'─' * 80}")
        
        # Timing from trace logs
        if stage.total_trace_time_us > 0:
            pct = (stage.total_trace_time_us / total_trace * 100) if total_trace > 0 else 0
            print(f"  Time: {format_duration(stage.total_trace_time_us)} ({pct:.1f}%)")
        
        if stage.total_gpu_time_us > 0:
            pct = (stage.total_gpu_time_us / total_gpu * 100) if total_gpu > 0 else 0
            print(f"  GPU Time: {format_duration(stage.total_gpu_time_us)} ({pct:.1f}%)")
        
        # CPU functions from trace (timing)
        if stage.trace_functions:
            agg = aggregate_trace(stage.trace_functions)
            print(f"\n  CPU Functions (timing from trace) - {len(agg)} functions:")
            for f in sorted(agg.values(), key=lambda x: -x.duration_us):
                cnt = f" (x{f.call_count})" if f.call_count > 1 else ""
                print(f"    • {f.name}{cnt}: {format_duration(f.duration_us)} ({f.file}:{f.line})")
        
        # CPU functions from VTune (identification) - aggregated by name
        if stage.vtune_functions:
            # Minimap2 source functions (most relevant - from minimap2/*/src)
            agg_mm2 = aggregate_vtune_minimap2(stage.vtune_functions)
            if agg_mm2:
                total_mm2_ms = sum(info['cpu_time_ms'] for info in agg_mm2.values())
                print(f"\n  Minimap2 Functions (VTune) - {len(agg_mm2)} functions, {format_duration(total_mm2_ms * 1000)} total:")
                for name, info in sorted(agg_mm2.items(), key=lambda x: -x[1]['cpu_time_ms']):
                    src = f" ({info['source_file']})" if info['source_file'] else ""
                    time_str = format_duration(info['cpu_time_ms'] * 1000)
                    cnt = f" (x{info['count']})" if info['count'] > 1 else ""
                    print(f"    • {name}{cnt}: {time_str}{src}")
            
            # External library functions (spdlog, STL, system, etc.) - show condensed
            agg_ext = aggregate_vtune_external(stage.vtune_functions)
            if agg_ext:
                total_ext_ms = sum(info['cpu_time_ms'] for info in agg_ext.values())
                sorted_ext = sorted(agg_ext.items(), key=lambda x: -x[1]['cpu_time_ms'])
                print(f"\n  External Functions (VTune) - {len(agg_ext)} functions, {format_duration(total_ext_ms * 1000)} total:")
                # Show top 5 external functions
                for name, info in sorted_ext[:5]:
                    src = f" ({info['source_file']})" if info['source_file'] else ""
                    time_str = format_duration(info['cpu_time_ms'] * 1000)
                    cnt = f" (x{info['count']})" if info['count'] > 1 else ""
                    print(f"    • {name}{cnt}: {time_str}{src}")
                if len(sorted_ext) > 5:
                    remaining = sum(info['cpu_time_ms'] for _, info in sorted_ext[5:])
                    print(f"    ... and {len(sorted_ext) - 5} more ({format_duration(remaining * 1000)})")
            
            # Unnamed/ROCm functions (system-generated names like func@0x..., func_NNN)
            agg_unnamed = aggregate_vtune_unnamed(stage.vtune_functions)
            if agg_unnamed:
                total_unnamed_ms = sum(info['cpu_time_ms'] for info in agg_unnamed.values())
                sorted_unnamed = sorted(agg_unnamed.items(), key=lambda x: -x[1]['cpu_time_ms'])
                print(f"\n  Unnamed/ROCm Functions ({len(agg_unnamed)} functions, {format_duration(total_unnamed_ms * 1000)} total):")
                # Show top 5 unnamed functions
                for name, info in sorted_unnamed[:5]:
                    time_str = format_duration(info['cpu_time_ms'] * 1000)
                    cnt = f" (x{info['count']})" if info['count'] > 1 else ""
                    print(f"    • {name}{cnt}: {time_str}")
                if len(sorted_unnamed) > 5:
                    remaining = sum(info['cpu_time_ms'] for _, info in sorted_unnamed[5:])
                    print(f"    ... and {len(sorted_unnamed) - 5} more ({format_duration(remaining * 1000)})")
        
        # GPU kernels
        if stage.gpu_kernels:
            agg = aggregate_gpu(stage.gpu_kernels)
            print(f"\n  GPU Kernels (identified by rocprofv3) - {len(agg)} kernels:")
            for name, info in sorted(agg.items(), key=lambda x: -x[1]['duration_ns']):
                cnt = f" (x{info['count']})" if info['count'] > 1 else ""
                print(f"    • {name}{cnt}: {format_duration(info['duration_ns']/1000)}")
        
        # GPU API calls (HIP/HSA)
        if stage.gpu_api_calls:
            agg = aggregate_gpu(stage.gpu_api_calls)
            print(f"\n  GPU API Calls (HIP/HSA) - {len(agg)} calls:")
            for name, info in sorted(agg.items(), key=lambda x: -x[1]['duration_ns']):
                cnt = f" (x{info['count']})" if info['count'] > 1 else ""
                print(f"    • {name}{cnt}: {format_duration(info['duration_ns']/1000)}")
        
        # GPU markers
        if stage.gpu_markers:
            agg = aggregate_gpu(stage.gpu_markers)
            print(f"\n  GPU Markers (ROCTX) - {len(agg)} markers:")
            for name, info in sorted(agg.items(), key=lambda x: -x[1]['duration_ns']):
                cnt = f" (x{info['count']})" if info['count'] > 1 else ""
                print(f"    • {name}{cnt}: {format_duration(info['duration_ns']/1000)}")
    
    # Hardware counter summary (if collected)
    if kernel_counters:
        print(f"\n{'─' * 80}")
        print("HARDWARE COUNTERS (rocprofv3 --pmc)")
        print(f"{'─' * 80}")
        print(f"  Kernels profiled: {len(kernel_counters)}")
        print()
        print(f"  {'Kernel':<40} {'VGPR':>5} {'SGPR':>5} {'LDS':>6} {'Scratch':>7} "
              f"{'VALU%':>6} {'VMEM%':>6} {'LDS%':>5} {'Occ%':>5} {'Cls':>8}")
        print(f"  {'-'*40} {'-----':>5} {'-----':>5} {'------':>6} {'-------':>7} "
              f"{'------':>6} {'------':>6} {'-----':>5} {'-----':>5} {'--------':>8}")
        for kname in sorted(kernel_counters,
                            key=lambda k: kernel_counters[k].get('dispatch_count', 0),
                            reverse=True):
            kc = kernel_counters[kname]
            res = kc.get('resources', {})
            der = kc.get('derived', {})
            short = kname[:37] + '...' if len(kname) > 40 else kname
            vgpr = str(res.get('Arch_VGPR', '-'))
            sgpr = str(res.get('SGPR', '-'))
            lds = str(res.get('LDS_Per_Workgroup', '-'))
            scratch = str(res.get('Scratch_Per_Workitem', '-'))
            valu = f"{der['valu_pct']:.0f}" if 'valu_pct' in der else '-'
            vmem = f"{der['vmem_pct']:.0f}" if 'vmem_pct' in der else '-'
            lds_pct = f"{der['lds_pct']:.0f}" if 'lds_pct' in der else '-'
            occ = f"{der['occupancy_pct']:.0f}" if 'occupancy_pct' in der else '-'
            cls = der.get('classification', '-')
            print(f"  {short:<40} {vgpr:>5} {sgpr:>5} {lds:>6} {scratch:>7} "
                  f"{valu:>6} {vmem:>6} {lds_pct:>5} {occ:>5} {cls:>8}")

        # Occupancy warnings
        low_occ = [(k, kc['derived']['occupancy_pct'], kc['derived'].get('occupancy_limiter', ''))
                    for k, kc in kernel_counters.items()
                    if kc.get('derived', {}).get('occupancy_pct', 100) < 50]
        if low_occ:
            print()
            print("  ⚠ Low Occupancy Warnings (< 50%):")
            for kname, occ_pct, limiter in sorted(low_occ, key=lambda x: x[1]):
                short = kname[:60] + '...' if len(kname) > 63 else kname
                print(f"    {short}: {occ_pct:.0f}% (limited by {limiter})")

        # LDS bank conflict warnings
        conflicts = [(k, kc['derived']['lds_conflict_rate'])
                     for k, kc in kernel_counters.items()
                     if kc.get('derived', {}).get('lds_conflict_rate', 0) > 5]
        if conflicts:
            print()
            print("  ⚠ LDS Bank Conflict Warnings:")
            for kname, rate in sorted(conflicts, key=lambda x: -x[1]):
                short = kname[:60] + '...' if len(kname) > 63 else kname
                print(f"    {short}: {rate:.1f}% conflict rate")
    
    print("\n" + "=" * 80)


# =============================================================================
# Main
# =============================================================================

def find_minimap2() -> Optional[str]:
    """Find minimap2 executable, preferring profile build for profiling."""
    paths = [
        # Profile build (has debug symbols + profiling instrumentation)
        Path(__file__).parent.parent / 'out' / 'profile' / 'build' / 'bin' / 'minimap2',
        # AMD default build
        Path(__file__).parent.parent / 'out' / 'amd-default' / 'build' / 'bin' / 'minimap2',
        # Generic build directory
        Path(__file__).parent.parent / 'build' / 'bin' / 'minimap2',
    ]
    for p in paths:
        if p.exists():
            return str(p)
    
    try:
        result = subprocess.run(['which', 'minimap2'], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass
    return None


def main():
    global MIN_DURATION_US

    parser = argparse.ArgumentParser(
        description='Analyze minimap2 pipeline stages',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--minimap2', '-m', help='Path to minimap2 executable')
    parser.add_argument('--output', '-o', help='Output directory')
    parser.add_argument('--format', '-f', choices=['text', 'json'], default='text')
    parser.add_argument('--no-vtune', action='store_true', help='Skip VTune profiling')
    parser.add_argument('--no-rocprof', action='store_true', help='Skip rocprofv3 profiling')
    parser.add_argument('--no-trace', action='store_true', help='Skip trace log timing')
    parser.add_argument('--reuse-data', action='store_true', 
                        help='Reuse existing profiling data from output directory (skip re-profiling)')
    parser.add_argument('--min-duration', type=float, default=MIN_DURATION_US,
                        help=f'Minimum duration threshold in µs (default: {MIN_DURATION_US})')
    parser.add_argument('--counters', action='store_true',
                        help='Collect hardware performance counters via rocprofv3 --pmc '
                             '(requires additional app re-runs)')
    parser.add_argument('mm_args', nargs='*', help='Minimap2 arguments (after --)')
    
    args = parser.parse_args()
    
    # Apply --min-duration globally so all aggregation functions respect it
    MIN_DURATION_US = args.min_duration
    
    # Find minimap2
    minimap2_path = args.minimap2 or find_minimap2()
    if not minimap2_path and not args.reuse_data:
        print("Error: minimap2 not found. Use --minimap2 to specify path.", file=sys.stderr)
        sys.exit(1)
    
    if not args.mm_args and not args.reuse_data:
        print("Error: No minimap2 arguments provided. Use: analyze_stages.py -- <minimap2 args>", file=sys.stderr)
        sys.exit(1)
    
    # Normalize arguments for profiling (force -t 1 and -a)
    original_mm_args = args.mm_args if args.mm_args else []
    mm_args = normalize_args_for_profiling(original_mm_args) if original_mm_args else []
    
    if minimap2_path:
        print(f"Minimap2: {minimap2_path}")
    if mm_args:
        print(f"Arguments: {' '.join(mm_args)}")
        # Check if -a was auto-added
        has_cigar = any(arg in ('-a', '-c', '--sam', '--cs') for arg in original_mm_args)
        if has_cigar:
            print(f"Note: Using -t 1 for accurate profiling")
        else:
            print(f"Note: Using -t 1 -a for accurate profiling (SAM output enables CIGAR alignment)")
    
    # Create or validate output directory
    if args.reuse_data and not args.output:
        print("Error: --reuse-data requires --output to specify existing data directory", file=sys.stderr)
        sys.exit(1)
    
    output_dir = args.output or tempfile.mkdtemp(prefix='mm2_analysis_')
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, 'logs')
    
    # Collect data
    trace_funcs: List[FunctionTiming] = []
    vtune_funcs: List[CPUFunction] = []
    vtune_tasks: Dict[str, List[CPUFunction]] = {}
    gpu_funcs: List[GPUFunction] = []
    gpu_memory: Dict[str, Any] = {}
    kernel_counters: Dict[str, Dict] = {}

    if args.reuse_data:
        print(f"Reusing data from: {output_dir}\n")
        # Preserve gpu_memory and kernel_counters from existing analysis.json
        # (memory sampling and counter collection don't run during --reuse-data)
        existing_json = os.path.join(output_dir, 'analysis.json')
        if os.path.exists(existing_json):
            try:
                with open(existing_json) as _f:
                    _old = json.load(_f)
                gpu_memory = _old.get('summary', {}).get('gpu_memory', {})
                kernel_counters = _old.get('kernel_counters', {})
                if gpu_memory:
                    print(f"  Loaded gpu_memory from existing analysis.json")
                if kernel_counters:
                    print(f"  Loaded kernel_counters from existing analysis.json")
            except Exception:
                pass
    else:
        print(f"Output: {output_dir}\n")

    # Start GPU memory sampling if rocm-smi is available
    mem_sampler: Optional[GpuMemorySampler] = None
    if not args.reuse_data and check_tool('rocm-smi'):
        mem_sampler = GpuMemorySampler()
        mem_sampler.snapshot_baseline()
        mem_sampler.start()
    
    # 1. Trace logs for timing
    if not args.no_trace:
        if args.reuse_data:
            trace_log_path = os.path.join(output_dir, 'trace.log')
            if os.path.exists(trace_log_path):
                print("[1/3] Reusing trace data from trace.log...")
                with open(trace_log_path) as f:
                    trace_output = f.read()
                trace_parser = TraceLogParser()
                trace_funcs = trace_parser.parse_output(trace_output)
                if trace_funcs:
                    print(f"      Found {len(trace_funcs)} top-level functions\n")
                else:
                    print("      No trace data found in trace.log\n")
            else:
                print("[1/3] Skipping trace collection (--reuse-data)")
                print("      Note: No trace.log found, run without --reuse-data for trace timing\n")
        else:
            print("[1/3] Collecting timing from trace logs...")
            trace_output = run_minimap2_trace(minimap2_path, mm_args, log_dir=output_dir)
            trace_parser = TraceLogParser()
            trace_funcs = trace_parser.parse_output(trace_output)
            if trace_funcs:
                print(f"      Found {len(trace_funcs)} top-level functions\n")
            else:
                print("      No trace data found. Ensure minimap2 was built with ENABLE_TRACE=ON\n")
    else:
        print("[1/3] Skipping trace (--no-trace)\n")
    
    # 2. VTune for CPU function identification
    if not args.no_vtune:
        vtune_dir = os.path.join(output_dir, 'vtune')
        
        if args.reuse_data and os.path.exists(vtune_dir):
            print("[2/3] Reusing VTune data...")
            vtune_parser = VTuneParser(vtune_dir)
            vtune_funcs = vtune_parser.parse_all()
            task_timing = vtune_parser.parse_sqlite_tasks()
            vtune_tasks = vtune_parser.parse_functions_by_task_timestamp()
            
            if vtune_tasks:
                task_count = sum(len(funcs) for name, funcs in vtune_tasks.items() if not name.startswith('__'))
                task_names = [name for name in vtune_tasks.keys() if not name.startswith('__')]
                print(f"      Found {task_count} function samples correlated with {len(task_names)} ITT tasks")
            print(f"      Found {len(vtune_funcs)} CPU functions\n")
        elif check_tool('vtune'):
            print("[2/3] Identifying CPU functions with VTune...")
            if run_vtune(minimap2_path, mm_args, vtune_dir, log_dir=log_dir):
                vtune_parser = VTuneParser(vtune_dir)
                vtune_funcs = vtune_parser.parse_all()
                
                # Parse SQLite for accurate ITT task timing
                task_timing = vtune_parser.parse_sqlite_tasks()
                
                # Parse functions correlated by timestamp with ITT tasks
                # This eliminates pattern-based hardcoding
                vtune_tasks = vtune_parser.parse_functions_by_task_timestamp()
                
                if vtune_tasks:
                    task_count = sum(len(funcs) for name, funcs in vtune_tasks.items() if not name.startswith('__'))
                    task_names = [name for name in vtune_tasks.keys() if not name.startswith('__')]
                    print(f"      Found {task_count} function samples correlated with {len(task_names)} ITT tasks")
                
                if task_timing:
                    print(f"      Found {len(vtune_funcs)} CPU functions, {len(task_timing)} ITT tasks (SQLite)")
                    # Show stage summary from ITT tasks
                    stage_times: Dict[str, float] = {}
                    for tt in task_timing:
                        stage_times[tt.stage] = stage_times.get(tt.stage, 0) + tt.duration_s
                    print(f"      ITT Stage Summary: " + ", ".join(f"{s}={t:.1f}s" for s, t in sorted(stage_times.items(), key=lambda x: -x[1])))
                else:
                    print(f"      Found {len(vtune_funcs)} CPU functions (no ITT task timing data)")
                print(f"      View detailed results: vtune-gui {vtune_dir}\n")
        else:
            print("[2/3] VTune not available, skipping CPU function identification\n")
    else:
        print("[2/3] Skipping VTune (--no-vtune)\n")
    
    # 3. rocprofv3 for GPU function identification
    if not args.no_rocprof:
        rocprof_dir = os.path.join(output_dir, 'rocprof')
        
        if args.reuse_data and os.path.exists(rocprof_dir):
            print("[3/3] Reusing rocprofv3 data...")
            rocprof_parser = RocprofParser(rocprof_dir)
            gpu_funcs = rocprof_parser.parse_all()
            total_calls = sum(f.call_count for f in gpu_funcs)
            print(f"      Found {total_calls:,} GPU function calls ({len(gpu_funcs)} unique)\n")
        elif check_tool('rocprofv3'):
            print("[3/3] Identifying GPU functions with rocprofv3...")
            if run_rocprofv3(minimap2_path, mm_args, rocprof_dir, log_dir=log_dir):
                rocprof_parser = RocprofParser(rocprof_dir)
                gpu_funcs = rocprof_parser.parse_all()
                total_calls = sum(f.call_count for f in gpu_funcs)
                print(f"      Found {total_calls:,} GPU function calls ({len(gpu_funcs)} unique)\n")
        else:
            print("[3/3] rocprofv3 not available, skipping GPU function identification\n")
    else:
        print("[3/3] Skipping rocprofv3 (--no-rocprof)\n")
    
    # 4. Hardware counter collection (optional, requires --counters)
    if args.counters and not args.no_rocprof:
        counter_dir = os.path.join(output_dir, 'rocprof_counters')

        if args.reuse_data and os.path.exists(counter_dir):
            print("[counters] Reusing counter data...")
            kernel_counters = parse_counter_results(counter_dir)
            print(f"      Parsed counters for {len(kernel_counters)} kernels\n")
        elif check_tool('rocprofv3'):
            print("[counters] Collecting hardware performance counters...")
            if run_rocprofv3_counters(minimap2_path, mm_args, counter_dir, log_dir=log_dir):
                kernel_counters = parse_counter_results(counter_dir)
                print(f"      Parsed counters for {len(kernel_counters)} kernels\n")
            else:
                print("      Counter collection failed\n")
        else:
            print("[counters] rocprofv3 not available\n")
    elif args.counters:
        print("[counters] Skipped (--no-rocprof disables GPU profiling)\n")

    # Stop GPU memory sampling
    if mem_sampler:
        gpu_memory = mem_sampler.stop()
        if gpu_memory:
            tracking = gpu_memory.get('tracking', 'system_wide')
            pp_count = gpu_memory.get('per_process_samples', 0)
            total_count = gpu_memory['sample_count']
            print(f"Memory Utilization ({total_count} samples, GPU tracking: {tracking}"
                  f"{f', {pp_count} per-process' if tracking == 'per_process' else ''}):")
            print(f"  GPU VRAM Total:    {format_bytes(gpu_memory['vram_total_bytes'])}")
            print(f"  GPU VRAM Baseline: {format_bytes(gpu_memory['vram_baseline_bytes'])}")
            print(f"  GPU VRAM Peak:     {format_bytes(gpu_memory['vram_peak_bytes'])}")
            print(f"  GPU VRAM Delta:    {format_bytes(gpu_memory['vram_delta_bytes'])}")
            rss_count = gpu_memory.get('cpu_rss_samples', 0)
            if rss_count:
                print(f"  CPU RSS Baseline:  {format_bytes(gpu_memory['cpu_rss_baseline_bytes'])}")
                print(f"  CPU RSS Peak:      {format_bytes(gpu_memory['cpu_rss_peak_bytes'])}")
                print(f"  CPU RSS Delta:     {format_bytes(gpu_memory['cpu_rss_delta_bytes'])}")
            print()

            # Generate memory timeline graph
            graph_path = os.path.join(output_dir, 'memory_timeline.png')
            mem_sampler.generate_graph(graph_path)

            # Strip timeseries from JSON output (keep stats only)
            gpu_memory.pop('timeseries', None)
    
    # Analyze
    analyzer = StageAnalyzer()
    analyzer.analyze(trace_funcs, vtune_funcs, gpu_funcs, vtune_tasks)
    
    # Compute kernel occupancy from rocprofv3 dispatch data
    kernel_occupancy = compute_kernel_occupancy(gpu_funcs) if gpu_funcs else []
    if kernel_occupancy:
        print(f"Computed occupancy for {len(kernel_occupancy)} GPU kernels")
    
    # Report
    print_report(analyzer, args.format, kernel_counters, gpu_memory)
    
    # Always save analysis.json (needed by generate_profiling_report.py regardless of --format)
    json_path = os.path.join(output_dir, 'analysis.json')
    report = _build_report_dict(
        analyzer, kernel_counters, gpu_memory,
        extra_summary={
            'trace_function_count': len(trace_funcs),
            'vtune_function_count': len(vtune_funcs),
            'gpu_function_count': sum(f.call_count for f in gpu_funcs),
        },
        kernel_occupancy=kernel_occupancy,
    )
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved to: {json_path}")
    if kernel_counters:
        print(f"  (includes hardware counters for {len(kernel_counters)} kernels)")


if __name__ == '__main__':
    main()
