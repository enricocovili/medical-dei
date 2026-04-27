#!/bin/bash

# --- Configuration ---
WORKSPACE_DIR="/workspace"
REPO_DIR="$WORKSPACE_DIR/sam3"
VENV_PATH="$REPO_DIR/.venv"

# --- Colors & UI Helpers ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

section() {
    clear
    echo -e "${BLUE}==============================================================${NC}"
    echo -e "${BLUE}  STEP: $1${NC}"
    echo -e "${BLUE}==============================================================${NC}"
    echo ""
}

notify_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}[SUCCESS] $1${NC}"
    else
        echo -e "${RED}[ERROR] $1 failed, but moving forward...${NC}"
    fi
    sleep 2
}

install_uv() {
    section "Installing uv Package Manager"
    if ! command -v uv &> /dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source $HOME/.cargo/env
        notify_status "uv installation"
    else
        echo -e "${GREEN}uv is already installed.${NC}"
    fi
}

clone_sam3_repo() {
    section "Cloning SAM 3 Repository"
    if [ ! -d "$REPO_DIR" ]; then
        git clone https://github.com/facebookresearch/sam3.git "$REPO_DIR"
        git -C "$REPO_DIR" checkout 86ed770
        notify_status "Repository cloning and checkout to 86ed770"
    else
        echo -e "${YELLOW}Directory $REPO_DIR already exists. Skipping clone.${NC}"
    fi
    cd "$REPO_DIR"
}

setup_venv() {
    section "Setting up Virtual Environment (Python 3.12)"
    if [ ! -d "$VENV_PATH" ]; then
        uv venv .venv --python 3.12
        notify_status "Virtual environment creation"
    else
        echo -e "${GREEN}Virtual environment already exists.${NC}"
    fi
    source .venv/bin/activate
}

detect_cuda_version() {
    local cuda_version_raw=""
    if command -v nvcc &> /dev/null; then
        cuda_version_raw="$(nvcc --version | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p' | head -n1)"
    elif command -v nvidia-smi &> /dev/null; then
        cuda_version_raw="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]\+\.[0-9]\+\).*/\1/p' | head -n1)"
    fi
    
    if [ -z "$cuda_version_raw" ]; then
        echo -e "${RED}CUDA not found. Aborting.${NC}"
        exit 1
    fi

    local major minor
    IFS='.' read -r major minor <<< "$cuda_version_raw"

    if [ -z "$major" ] || [ -z "$minor" ]; then
        echo -e "${RED}Invalid CUDA version format: ${cuda_version_raw}. Aborting.${NC}"
        exit 1
    fi

    if [ "$major" -lt 12 ] || { [ "$major" -eq 12 ] && [ "$minor" -lt 6 ]; }; then
        echo -e "${RED}Detected CUDA ${cuda_version_raw}, but 12.6+ is required. Aborting.${NC}"
        exit 2
    fi

    echo "cu${cuda_version_raw/./}"
}

install_dependencies() {
    section "Installing PyTorch & Heavy Kernels"
    echo -e "${YELLOW}This may take a few minutes depending on host bandwidth...${NC}"
    
    local cuda_tag=$(detect_cuda_version)
    local pytorch_index_url="https://download.pytorch.org/whl/${cuda_tag}"

    echo -e "${BLUE}Using CUDA tag: ${cuda_tag}${NC}"
    echo -e "${BLUE}Using index: ${pytorch_index_url}${NC}"
    
    uv pip install torch==2.10.0 torchvision --index-url "${pytorch_index_url}"
    notify_status "PyTorch installation"
    
    uv pip install -e .
    notify_status "SAM 3 library installation"
    
    uv pip install einops ninja pycocotools psutil opencv-python pillow easyocr
    notify_status "Utility libraries installation"
    
    uv pip install flash-attn-3 --no-deps --index-url "${pytorch_index_url}"
    notify_status "Flash Attention 3 installation"
    
    uv pip install git+https://github.com/ronghanghu/cc_torch.git
    notify_status "CC_Torch installation"
}

hf_authentication() {
    section "Hugging Face Authentication"
    if [ -z "$HF_TOKEN" ]; then
        echo -e "${YELLOW}HF_TOKEN not found in environment variables.${NC}"
        echo -e "${BLUE}Starting interactive login...${NC}"
        hf auth login
    fi
    notify_status "Hugging Face authentication"
}

print_summary() {
    clear
    echo -e "${GREEN}==============================================================${NC}"
    echo -e "${GREEN}             SETUP PROCESS COMPLETE                          ${NC}"
    echo -e "${GREEN}==============================================================${NC}"
    echo -e "Repo: ${REPO_DIR}"
    echo -e "Venv: ${VENV_PATH}"
    echo ""
    echo -e "${YELLOW}To start working, run:${NC}"
    echo -e "${BLUE}cd $REPO_DIR && source .venv/bin/activate${NC}"
    echo -e "${GREEN}==============================================================${NC}"
}

# --- Main Execution ---
install_uv
clone_sam3_repo
setup_venv
install_dependencies
hf_authentication
print_summary

# detect_cuda_version
# install_dependencies
