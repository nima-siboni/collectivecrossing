#!/bin/bash

# Documentation management script for Collective Crossing

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  serve     Start local development server"
    echo "  build     Build the documentation"
    echo "  clean     Clean the build directory"
    echo "  deploy    Build and deploy to GitHub Pages"
    echo "  check     Check for broken links"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 serve     # Start local server at http://localhost:8000"
    echo "  $0 build     # Build documentation to site/ directory"
    echo "  $0 clean     # Remove site/ directory"
}

# Function to check if uv is available
check_uv() {
    if ! command -v uv &> /dev/null; then
        print_error "uv is not installed. Please install it first."
        print_status "Visit: https://docs.astral.sh/uv/"
        exit 1
    fi
}

# Function to check if mkdocs is available
check_mkdocs() {
    if ! uv run mkdocs --version &> /dev/null; then
        print_error "MkDocs is not installed. Installing now..."
        uv add --dev mkdocs mkdocs-material
    fi
}

# Function to serve documentation locally
serve_docs() {
    print_status "Starting local development server..."
    print_status "Documentation will be available at: http://localhost:8000"
    print_status "Press Ctrl+C to stop the server"
    echo ""
    uv run mkdocs serve --dev-addr=127.0.0.1:8000
}

# Function to build documentation
build_docs() {
    print_status "Building documentation..."
    uv run mkdocs build
    print_success "Documentation built successfully in site/ directory"
}

# Function to clean build directory
clean_docs() {
    print_status "Cleaning build directory..."
    if [ -d "site" ]; then
        rm -rf site
        print_success "Build directory cleaned"
    else
        print_warning "No build directory found"
    fi
}

# Function to deploy documentation
deploy_docs() {
    print_status "Building documentation for deployment..."
    uv run mkdocs build
    print_success "Documentation built successfully"
    print_status "Deploying to GitHub Pages..."
    uv run mkdocs gh-deploy
    print_success "Documentation deployed successfully!"
    print_status "Your docs are available at: https://nima-siboni.github.io/collectivecrossing/"
}

# Function to check for broken links
check_links() {
    print_status "Checking for broken links..."
    if command -v linkchecker &> /dev/null; then
        uv run mkdocs build
        linkchecker site/index.html
    else
        print_warning "linkchecker not found. Install it with: pip install linkchecker"
        print_status "You can also manually check the built site in the site/ directory"
    fi
}

# Main script logic
main() {
    # Check dependencies
    check_uv
    check_mkdocs

    # Parse command
    case "${1:-help}" in
        serve)
            serve_docs
            ;;
        build)
            build_docs
            ;;
        clean)
            clean_docs
            ;;
        deploy)
            deploy_docs
            ;;
        check)
            check_links
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            print_error "Unknown command: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
