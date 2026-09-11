#!/bin/bash
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
set -e

# Create .bashrc that sources /host_home/.bashrc
echo '[ -f /host_home/.bashrc ] && . /host_home/.bashrc' > ~/.bashrc

# Run host-specific post-create steps if present (untracked, per-user)
LOCAL_SCRIPT="$(dirname "$0")/localPostCreateCommand.sh"
if [ -x "$LOCAL_SCRIPT" ]; then
    "$LOCAL_SCRIPT"
elif [ -f "$LOCAL_SCRIPT" ]; then
    bash "$LOCAL_SCRIPT"
fi
