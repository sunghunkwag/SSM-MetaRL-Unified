# 🚀 Deployment Guide for SSM-MetaRL-Unified

This guide will help you deploy the package to PyPI and resolve the PyPI badge version issue.

## Step 1: Setup PyPI Account and API Token

### 1.1 Create PyPI Account
1. Go to [https://pypi.org/account/register/](https://pypi.org/account/register/)
2. Create an account or login to existing account
3. Verify your email address

### 1.2 Generate API Token
1. Go to [https://pypi.org/manage/account/](https://pypi.org/manage/account/)
2. Scroll down to "API tokens" section
3. Click "Add API token"
4. Token name: `SSM-MetaRL-Unified`
5. Scope: "Entire account" (or specific to this project after first upload)
6. Copy the generated token (starts with `pypi-`)

### 1.3 Optional: Test PyPI Setup
1. Go to [https://test.pypi.org/account/register/](https://test.pypi.org/account/register/)
2. Create account and generate token for testing

## Step 2: Configure GitHub Secrets

### 2.1 Access Repository Settings
1. Go to your repository: https://github.com/sunghunkwag/SSM-MetaRL-Unified
2. Click "Settings" tab
3. In left sidebar, click "Secrets and variables" → "Actions"

### 2.2 Add Required Secrets
Click "New repository secret" for each:

**Required:**
- **Name**: `PYPI_API_TOKEN`
- **Value**: Your PyPI API token (starts with `pypi-`)

**Optional (for testing):**
- **Name**: `TEST_PYPI_API_TOKEN` 
- **Value**: Your Test PyPI API token

## Step 3: Deploy to PyPI

### Option A: Automatic Deployment (Recommended)

1. **Create and push a version tag:**
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```

2. **Monitor GitHub Actions:**
   - Go to "Actions" tab in your repository
   - Watch the "Publish to PyPI" workflow
   - It should automatically build and upload to PyPI

### Option B: Manual Deployment

1. **Trigger manual workflow:**
   - Go to "Actions" tab
   - Click "Publish to PyPI" workflow
   - Click "Run workflow" button
   - This will deploy to Test PyPI first

2. **Verify test deployment:**
   ```bash
   pip install -i https://test.pypi.org/simple/ ssm-metarl-unified
   python -c "import ssm_metarl_unified; print('Success!')"
   ```

3. **Deploy to production PyPI:**
   - Create a release tag as in Option A

## Step 4: Verify Deployment

### 4.1 Check PyPI Page
- Visit: https://pypi.org/project/ssm-metarl-unified/
- Should show version 1.0.0 and package details

### 4.2 Test Installation
```bash
# Install from PyPI
pip install ssm-metarl-unified

# Test import
python -c "from ssm_metarl_unified import StateSpaceModel; print('✅ Import successful!')"

# Test CLI tools
ssm-metarl-train --help
ssm-metarl-benchmark --help
ssm-metarl-test
```

### 4.3 Verify Badge
- The PyPI badge in README.md should now show version "1.0.0"
- Badge URL: `https://img.shields.io/pypi/v/ssm-metarl-unified`
- May take 5-10 minutes for badge cache to update

## Step 5: Future Updates

For subsequent releases:

1. **Update version in pyproject.toml:**
   ```toml
   version = "1.0.1"
   ```

2. **Update version in __init__.py:**
   ```python
   __version__ = "1.0.1"
   ```

3. **Create new tag:**
   ```bash
   git add .
   git commit -m "Release v1.0.1"
   git tag v1.0.1
   git push origin v1.0.1
   ```

## Troubleshooting

### Common Issues:

1. **"Package already exists" error:**
   - Package name is taken
   - Choose a different name in pyproject.toml

2. **Authentication failed:**
   - Check API token is correct
   - Ensure token has proper permissions
   - Check GitHub Secrets are set correctly

3. **Badge still shows no version:**
   - Wait 5-10 minutes for cache refresh
   - Check package is actually on PyPI
   - Verify badge URL matches package name

4. **Import errors after installation:**
   - Check package structure in pyproject.toml
   - Verify __init__.py files exist
   - Test with `pip install -e .` locally first

### Quick Commands Reference:

```bash
# Local testing before deployment
python -m build
twine check dist/*

# Manual upload to Test PyPI
twine upload --repository testpypi dist/*

# Manual upload to PyPI
twine upload dist/*

# Clean build artifacts
rm -rf dist/ build/ *.egg-info/
```

## Support

If you encounter issues:
1. Check GitHub Actions logs
2. Verify PyPI package page
3. Test local installation
4. Check this guide for common solutions

---

**🎉 Once deployed successfully, your package will be available worldwide via `pip install ssm-metarl-unified`!**