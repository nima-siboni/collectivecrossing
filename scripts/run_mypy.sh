#!/bin/bash
# Mypy type checking script for the collectivecrossing project

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_ROOT/src"
TEST_DIR="$PROJECT_ROOT/tests"

echo -e "${BLUE}🔍 Running mypy type checking...${NC}"
echo -e "${BLUE}📁 Project root: $PROJECT_ROOT${NC}"
echo -e "${BLUE}📁 Source directory: $SRC_DIR${NC}"
echo "--------------------------------------------------"

# Build the mypy command using uv
MYPY_CMD="uv run mypy --config-file $PROJECT_ROOT/pyproject.toml $SRC_DIR"

# Add test directory if it exists
if [ -d "$TEST_DIR" ]; then
    MYPY_CMD="$MYPY_CMD $TEST_DIR"
fi

echo -e "${BLUE}🚀 Command: $MYPY_CMD${NC}"
echo

# Run mypy
cd "$PROJECT_ROOT"
if eval "$MYPY_CMD"; then
    echo "--------------------------------------------------"
    echo -e "${GREEN}✅ Mypy type checking passed!${NC}"
    exit 0
else
    echo "--------------------------------------------------"
    echo -e "${RED}❌ Mypy type checking failed!${NC}"
    exit 1
fi
