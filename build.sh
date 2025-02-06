#!/bin/bash

# Strict mode settings
set -euo pipefail
IFS=$'\n\t'

# Script variables
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly ROOT_DIR="$(pwd)"
readonly GIT_DIR="$(realpath "$ROOT_DIR/..")"
readonly FIFTYONE_DIR="$GIT_DIR/fiftyone"
readonly API_DOC_DIR="$GIT_DIR/api_docs"
readonly TS_API_DOC_DIR="$GIT_DIR/ts_api_docs"
readonly VENV_ACTIVATE="/home/spousty/virtualenvs/vdoc-mkdocs/bin/activate"
readonly GIT_BRANCH="main"  # Default branch name
readonly NODE_VERSION=17.9.0

# Color definitions for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

# Function to display script usage
show_usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Build documentation for the FiftyOne project.

Options:
    -h, --help              Show this help message
    -v, --verbose          Enable verbose output
    --skip-git             Skip git pull operations
    --skip-python-api      Skip Python API documentation build
    --skip-ts-api          Skip TypeScript API documentation build
    --skip-all-api         Skip all API documentation builds
    --version VERSION      Specify the version number (default: 1.3)
    --branch BRANCH       Specify the git branch (default: main)
EOF
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check dependencies
check_dependencies() {
    local missing_deps=()

    # Check for git
    if ! command_exists git; then
        missing_deps+=("git")
    fi

    # Check for python dependencies
    if ! command_exists python3; then
        missing_deps+=("python3")
    fi

    if ! command_exists pydoctor; then
        missing_deps+=("pydoctor")
    fi

    if ! command_exists mkdocs; then
        missing_deps+=("mkdocs")
    fi

    # Check for Node.js and npm
    if ! command_exists node; then
        missing_deps+=("node")
    fi

    if ! command_exists npm; then
        missing_deps+=("npm")
    fi

    # If there are missing dependencies, print them and exit
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "Missing required dependencies:"
        printf '%s\n' "${missing_deps[@]}"
        log_error "Please install missing dependencies and try again."
        exit 1
    fi
}

# Function to check if directory exists and is a git repository
check_git_repo() {
    local dir=$1
    if [ ! -d "$dir" ]; then
        log_error "Directory $dir does not exist"
        exit 1
    fi
    if [ ! -d "$dir/.git" ]; then
        log_error "Directory $dir is not a git repository"
        exit 1
    fi
}

# Function to handle errors
handle_error() {
    local line_no=$1
    local last_cmd=$2
    {
        echo "----------------------------------------"
        echo "Error occurred at $(date '+%Y-%m-%d %H:%M:%S')"
        echo "Line number: $line_no"
        echo "Last command: $last_cmd"
        echo "Current directory: $(pwd)"
        echo "Environment:"
        env | grep -v PASSWORD
        echo "----------------------------------------"
        if [ $VERBOSE -eq 1 ]; then
            echo "Call stack:"
            local frame=0
            while caller $frame; do
                ((frame++))
            done
            echo "----------------------------------------"
        fi
    } >> /tmp/build_docs_error.log
    exit 1
}

# Function to clean up on exit
cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        log_error "Script failed with exit code $exit_code"
        if [ -f /tmp/build_docs_error.log ]; then
            log_error "Error details:"
            cat /tmp/build_docs_error.log >&2
            echo "----------------------------------------" >&2
            echo "Full environment state at exit:" >&2
            printenv >&2
            rm -f /tmp/build_docs_error.log
        fi
    fi
    exit $exit_code
}

# Function to check and activate virtual environment
check_venv() {
    if [ ! -f "$VENV_ACTIVATE" ]; then
        log_error "Virtual environment activation script not found at: $VENV_ACTIVATE"
        exit 1
    fi
    log_info "Activating virtual environment..."
    # shellcheck source=/dev/null
    source "$VENV_ACTIVATE" || {
        log_error "Failed to activate virtual environment"
        exit 1
    }
}

# Function to safely switch git branch
switch_git_branch() {
    local target_branch=$1
    local current_branch

    # Get current branch
    current_branch=$(git symbolic-ref --short HEAD 2>/dev/null || git rev-parse --short HEAD)

    if [ "$current_branch" != "$target_branch" ]; then
        log_info "Switching from branch $current_branch to $target_branch..."

        # Check if the branch exists locally
        if ! git show-ref --verify --quiet "refs/heads/$target_branch"; then
            # Check if branch exists on remote
            if git fetch origin "$target_branch" 2>/dev/null; then
                git checkout -b "$target_branch" "origin/$target_branch" || {
                    log_error "Failed to checkout branch $target_branch"
                    exit 1
                }
            else
                log_error "Branch $target_branch not found locally or in remote"
                exit 1
            fi
        else
            git checkout "$target_branch" || {
                log_error "Failed to switch to branch $target_branch"
                exit 1
            }
        fi
    fi
}

# Set up error handling
trap 'handle_error ${LINENO} "${BASH_COMMAND}"' ERR
trap cleanup EXIT

# Ensure error log is clean at start
rm -f /tmp/build_docs_error.log

# Set up error logging
exec 5>&2
exec 2> >(tee /tmp/build_docs_error.log >&5)

# Parse command line arguments
VERBOSE=0
SKIP_GIT=0
SKIP_PYTHON_API=0
SKIP_TS_API=0
VERSION="1.3"
BRANCH="$GIT_BRANCH"

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_usage
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=1
            set -x  # Enable bash debug mode
            shift
            ;;
        --skip-git)
            SKIP_GIT=1
            shift
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --branch)
            BRANCH="$2"
            shift 2
            ;;
        --skip-python-api)
            SKIP_PYTHON_API=1
            shift
            ;;
        --skip-ts-api)
            SKIP_TS_API=1
            shift
            ;;
        --skip-all-api)
            SKIP_PYTHON_API=1
            SKIP_TS_API=1
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution starts here
main() {
    log_info "Starting documentation build process..."

    # Check and activate virtual environment
    check_venv

    # Check dependencies
    log_info "Checking dependencies..."
    check_dependencies

    # Verify git repositories
    check_git_repo "$FIFTYONE_DIR"
    check_git_repo "$ROOT_DIR"

    # Update repositories if not skipped
    if [ $SKIP_GIT -eq 0 ]; then
        log_info "Updating fiftyone repository..."
        cd "$FIFTYONE_DIR" || exit 1
        switch_git_branch "$BRANCH"
        git pull origin "$BRANCH" || {
            log_error "Failed to pull latest changes from branch $BRANCH"
            exit 1
        }

        cd "$ROOT_DIR" || exit 1
        log_info "Updating documentation repository..."
        git pull origin "$BRANCH" || {
            log_error "Failed to pull latest changes for documentation"
            exit 1
        }
    fi

    # Create output directories if they don't exist
    log_info "Creating output directories..."
    mkdir -p "$API_DOC_DIR"
    mkdir -p "$TS_API_DOC_DIR"

    # Build Python API documentation
    if [ $SKIP_PYTHON_API -eq 0 ]; then
        log_info "Building Python API documentation with pydoctor..."
        cd "$FIFTYONE_DIR" || exit 1

        # Ensure we're in the correct directory and fiftyone subdirectory exists
        if [ ! -d "fiftyone" ]; then
            log_error "Python source directory 'fiftyone' not found in $FIFTYONE_DIR"
            log_error "Current directory: $(pwd)"
            log_error "Directory contents:"
            ls -la
            exit 1
        fi

        mkdir -p "$API_DOC_DIR"

        pydoctor \
            --project-name=FiftyOne \
            --project-version="$VERSION" \
            --project-url=https://github.com/voxel51/ \
            --html-viewsource-base="https://github.com/voxel51/fiftyone/tree/v${VERSION}" \
            --html-base-url=https://docs.voxel51.com/api \
            --html-output="$API_DOC_DIR" \
            --docformat=google \
            --intersphinx=https://docs.python.org/3/objects.inv \
            fiftyone
        log_info "Finished the pydoctor build"
    else
        log_info "Skipping Python API documentation build..."
    fi

    # Build TypeScript API documentation
    if [ $SKIP_TS_API -eq 0 ]; then
        log_info "Building TypeScript API documentation..."
        cd "$FIFTYONE_DIR/app" || exit 1
        nvm use ${NODE_VERSION}
        yarn install > /dev/null 2>&1
        yarn workspace @fiftyone/fiftyone compile
        NODE_OPTIONS=--max-old-space-size=4096 && tsc && vite build
        npx typedoc \
            --out "$TS_API_DOC_DIR" \
            --name "FiftyOne TypeScript API" \
            --options typedoc.js \
            --theme default \
    else
        log_info "Skipping TypeScript API documentation build..."
    fi

    # Return to root directory and set up symlinks
    cd "$ROOT_DIR" || exit 1

    log_info "Creating symlinks..."
    # Handle symlinks based on what was built
    if [ $SKIP_PYTHON_API -eq 0 ]; then
        if [ -L docs/api ]; then
            rm docs/api
        fi
        ln -s "$API_DOC_DIR" docs/api
    fi

    if [ $SKIP_TS_API -eq 0 ]; then
        if [ -L docs/ts-api ]; then
            rm docs/ts-api
        fi
        ln -s "$TS_API_DOC_DIR" docs/ts-api
    fi

    # Build final documentation
    log_info "Building documentation with mkdocs..."
    mkdocs build

    log_info "Documentation build complete!"
}

# Execute main function
main "$@"