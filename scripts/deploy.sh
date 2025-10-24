#!/bin/bash
# Quick deployment script for SSM-MetaRL-Unified
# This script helps automate the PyPI deployment process

set -e  # Exit on any error

echo "🚀 SSM-MetaRL-Unified Deployment Script"
echo "==========================================="

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

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    print_error "pyproject.toml not found. Are you in the project root?"
    exit 1
fi

# Get current version from pyproject.toml
CURRENT_VERSION=$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')
print_status "Current version: $CURRENT_VERSION"

# Ask user what they want to do
echo ""
echo "What would you like to do?"
echo "1) Deploy current version ($CURRENT_VERSION) to PyPI"
echo "2) Create new version and deploy"
echo "3) Test deployment to Test PyPI"
echo "4) Check deployment status"
echo "5) Exit"
echo ""
read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        print_status "Deploying current version $CURRENT_VERSION..."
        
        # Check if tag already exists
        if git rev-parse "v$CURRENT_VERSION" >/dev/null 2>&1; then
            print_warning "Tag v$CURRENT_VERSION already exists!"
            read -p "Do you want to delete and recreate it? (y/N): " confirm
            if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
                git tag -d "v$CURRENT_VERSION"
                git push origin --delete "v$CURRENT_VERSION" 2>/dev/null || true
            else
                print_error "Deployment cancelled."
                exit 1
            fi
        fi
        
        # Create and push tag
        print_status "Creating tag v$CURRENT_VERSION..."
        git tag "v$CURRENT_VERSION"
        git push origin "v$CURRENT_VERSION"
        
        print_success "Tag created and pushed! GitHub Actions will handle the deployment."
        print_status "Monitor progress at: https://github.com/sunghunkwag/SSM-MetaRL-Unified/actions"
        ;;
        
    2)
        print_status "Current version: $CURRENT_VERSION"
        read -p "Enter new version (e.g., 1.0.1): " NEW_VERSION
        
        if [ -z "$NEW_VERSION" ]; then
            print_error "Version cannot be empty!"
            exit 1
        fi
        
        print_status "Updating version to $NEW_VERSION..."
        
        # Update pyproject.toml
        sed -i.bak "s/version = \"$CURRENT_VERSION\"/version = \"$NEW_VERSION\"/" pyproject.toml
        
        # Update __init__.py
        sed -i.bak "s/__version__ = \"$CURRENT_VERSION\"/__version__ = \"$NEW_VERSION\"/" __init__.py
        
        # Clean up backup files
        rm -f pyproject.toml.bak __init__.py.bak
        
        print_success "Version updated to $NEW_VERSION"
        
        # Ask if user wants to commit and deploy
        read -p "Commit changes and deploy? (Y/n): " commit_confirm
        if [[ $commit_confirm != [nN] && $commit_confirm != [nN][oO] ]]; then
            git add pyproject.toml __init__.py
            git commit -m "Release v$NEW_VERSION"
            git tag "v$NEW_VERSION"
            git push origin main  # or master, depending on your default branch
            git push origin "v$NEW_VERSION"
            
            print_success "Version $NEW_VERSION deployed!"
            print_status "Monitor progress at: https://github.com/sunghunkwag/SSM-MetaRL-Unified/actions"
        fi
        ;;
        
    3)
        print_status "Testing deployment to Test PyPI..."
        print_warning "This requires TEST_PYPI_API_TOKEN to be set in GitHub Secrets."
        
        # Trigger manual workflow
        print_status "Go to GitHub Actions and manually trigger the 'Publish to PyPI' workflow."
        print_status "URL: https://github.com/sunghunkwag/SSM-MetaRL-Unified/actions/workflows/publish.yml"
        ;;
        
    4)
        print_status "Checking deployment status..."
        
        # Check if package exists on PyPI
        if curl -s "https://pypi.org/pypi/ssm-metarl-unified/json" > /dev/null 2>&1; then
            PYPI_VERSION=$(curl -s "https://pypi.org/pypi/ssm-metarl-unified/json" | python3 -c "import sys, json; print(json.load(sys.stdin)['info']['version'])")
            print_success "Package is available on PyPI!"
            print_status "PyPI version: $PYPI_VERSION"
            print_status "Local version: $CURRENT_VERSION"
            
            if [ "$PYPI_VERSION" = "$CURRENT_VERSION" ]; then
                print_success "Versions match! ✅"
            else
                print_warning "Version mismatch! Local version is newer." 
            fi
            
            print_status "Package URL: https://pypi.org/project/ssm-metarl-unified/"
            print_status "Install with: pip install ssm-metarl-unified"
        else
            print_warning "Package not found on PyPI yet."
            print_status "Check GitHub Actions: https://github.com/sunghunkwag/SSM-MetaRL-Unified/actions"
        fi
        ;;
        
    5)
        print_status "Goodbye! 👋"
        exit 0
        ;;
        
    *)
        print_error "Invalid choice. Please run the script again."
        exit 1
        ;;
esac

echo ""
print_success "Deployment script completed! 🎉"
echo ""
echo "Next steps:"
echo "1. Monitor GitHub Actions for build status"
echo "2. Check PyPI page once deployed: https://pypi.org/project/ssm-metarl-unified/"
echo "3. Test installation: pip install ssm-metarl-unified"
echo "4. The PyPI badge should update automatically within 5-10 minutes"
echo ""