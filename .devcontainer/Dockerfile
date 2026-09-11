# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
ARG BASE_IMAGE=rocm/dev-ubuntu-24.04:7.2-complete
FROM ${BASE_IMAGE}

# Install compilers & dependencies for minimap2.
# Extra apt repos: Kitware (recent CMake) and apt.llvm.org for llvm-22. The
# ROCm base image ships AMD clang 22 (used by the coverage-clang build) but no
# llvm-cov/llvm-profdata, and Ubuntu's stock llvm is v18 which cannot read the
# v22 raw profile format. llvm-22 provides matching llvm-cov-22/llvm-profdata-22
# (picked up automatically by scripts/coverage.sh).
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common lsb-release gnupg wget ca-certificates && \
    wget -O- https://apt.kitware.com/keys/kitware-archive-latest.asc | \
        gpg --dearmor | tee /usr/share/keyrings/kitware-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" \
        > /etc/apt/sources.list.d/kitware.list && \
    wget -O- https://apt.llvm.org/llvm-snapshot.gpg.key | \
        gpg --dearmor | tee /usr/share/keyrings/llvm-archive-keyring.gpg > /dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/llvm-archive-keyring.gpg] https://apt.llvm.org/$(lsb_release -cs)/ llvm-toolchain-$(lsb_release -cs)-22 main" \
        > /etc/apt/sources.list.d/llvm-22.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential git gcc g++ cmake \
        ninja-build yasm python3-venv python3-pip pipx \
        openssh-client sudo \
        zlib1g-dev \
        clang llvm libclang-rt-dev clang-format clang-tidy clang-tools \
        llvm-22 \
        gcovr samtools && \
    rm -rf /var/lib/apt/lists/*

# Install k8 JavaScript shell (required for paftools.js and mmphase.js)
RUN curl -L https://github.com/attractivechaos/k8/releases/download/v1.2/k8-1.2.tar.bz2 | tar -jxf - \
    && cp k8-1.2/k8-x86_64-Linux /usr/local/bin/k8 \
    && chmod +x /usr/local/bin/k8 \
    && rm -rf k8-1.2

# Install Python tools for coverage (gcovr, lcov_cobertura) system-wide so
# they land on PATH for every user (default pipx locations go into the
# invoking user's $HOME, which would hide them from the dev user).
ENV PIPX_HOME=/opt/pipx \
    PIPX_BIN_DIR=/usr/local/bin
RUN pipx install gcovr && \
    pipx install lcov_cobertura

# Create a shared Python virtualenv with the project dev/test dependencies
# (cython, pytest, pytest-cov, coverage, build tooling) plus the helper-script
# dependencies (numpy, matplotlib, requests, tqdm, zstandard, flatbuffers).
# Living at /opt/venv it survives the workspace bind-mount (a workspace-local
# .venv would be shadowed at runtime) and its console scripts (cython, pytest)
# have correct shebangs. Putting /opt/venv/bin on PATH lets CMake find cython
# and scripts/coverage.sh + the scripts/ tooling find python/pytest without any
# per-user setup.
COPY requirements-dev.txt /tmp/requirements-dev.txt
COPY scripts/requirements.txt /tmp/scripts-requirements.txt
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir \
        -r /tmp/requirements-dev.txt \
        -r /tmp/scripts-requirements.txt && \
    rm -f /tmp/requirements-dev.txt /tmp/scripts-requirements.txt
ENV PATH="/opt/venv/bin:${PATH}"

COPY certs/* /usr/local/share/ca-certificates/
RUN update-ca-certificates

# Add ROCm to PATH
ENV PATH="/opt/rocm/bin:${PATH}"

ARG USER_UID=1001
ARG USER_GID=1001
ARG USERNAME=user
ARG RENDER_GID=109
ARG VIDEO_GID=44

# Set HOME and configure shell
ENV HOME=/home/$USERNAME
WORKDIR ${HOME}

# Create user with matching UID/GID (sudo comes from the main apt layer)
RUN groupadd --gid ${USER_GID} ${USERNAME} || true && \
    groupmod -g ${RENDER_GID} render || true && \
    groupmod -g ${VIDEO_GID} video || true && \
    useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} -s /bin/bash && \
    usermod -aG ${VIDEO_GID},${RENDER_GID} ${USERNAME} && \
    echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} && \
    chmod 0440 /etc/sudoers.d/${USERNAME} && \
    passwd -d ${USERNAME} && \
    chown -R ${USER_UID}:${USER_GID} ${HOME} && \
    chown -R ${USER_UID}:${USER_GID} /opt/venv

USER ${USERNAME}