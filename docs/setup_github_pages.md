# Setting Up GitHub Pages with MkDocs

This guide will help you set up GitHub Pages to host your Collective Crossing documentation using MkDocs.

## 🚀 Quick Setup

### 1. **Enable GitHub Pages**

1. Go to your repository on GitHub: `https://github.com/nima-siboni/collectivecrossing`
2. Click on **Settings** tab
3. Scroll down to **Pages** section in the left sidebar
4. Under **Source**, select **GitHub Actions**
5. Click **Save**

### 2. **Configure GitHub Actions**

The GitHub Actions workflow is already configured in `.github/workflows/docs.yml`. This workflow will:

- Build the documentation when you push to the `main` branch
- Deploy it to GitHub Pages automatically
- Handle pull requests for testing

### 3. **First Deployment**

1. Commit and push your changes:
   ```bash
   git add .
   git commit -m "Add MkDocs documentation setup"
   git push origin main
   ```

2. Go to the **Actions** tab on GitHub to monitor the deployment
3. Once complete, your documentation will be available at:
   `https://nima-siboni.github.io/collectivecrossing/`

## 🛠️ Local Development

### Using the Documentation Script

We've created a convenient script to manage documentation:

```bash
# Start local development server
./scripts/docs.sh serve

# Build documentation
./scripts/docs.sh build

# Clean build directory
./scripts/docs.sh clean

# Check for broken links
./scripts/docs.sh check

# Show help
./scripts/docs.sh help
```

### Manual Commands

```bash
# Install dependencies (if not already done)
uv sync --dev

# Start local server
uv run mkdocs serve

# Build documentation
uv run mkdocs build

# Deploy (builds and provides deployment instructions)
uv run mkdocs gh-deploy
```

## 📁 Project Structure

```
collectivecrossing/
├── docs/                          # Documentation source files
│   ├── index.md                   # Home page
│   ├── installation.md            # Installation guide
│   ├── usage.md                   # Usage guide
│   ├── development.md             # Development guide
│   ├── features.md                # Features overview
│   ├── setup_github_pages.md      # This guide
│   └── assets/                    # Images and other assets
├── mkdocs.yml                     # MkDocs configuration
├── .github/workflows/docs.yml     # GitHub Actions workflow
└── scripts/docs.sh               # Documentation management script
```

## ⚙️ Configuration

### MkDocs Configuration (`mkdocs.yml`)

The configuration uses minimal settings with Material theme defaults:

- **Material theme** with default styling
- **Navigation structure** for organized content
- **Search functionality** for finding content
- **Responsive design** for mobile devices
- **Clean and simple** appearance

## 🔧 Customization

### Adding New Pages

1. Create a new `.md` file in the `docs/` directory
2. Add it to the navigation in `mkdocs.yml`:

```yaml
nav:
  - Home: index.md
  - Getting Started:
    - Installation: installation.md
    - Usage Guide: usage.md
    - Your New Page: your_new_page.md  # Add this line
```

### Changing the Theme

You can customize the Material theme in `mkdocs.yml`:

```yaml
theme:
  name: material
  palette:
    - scheme: default
      primary: indigo  # Change color here
      accent: indigo
```

### Adding Plugins

To add new MkDocs plugins:

1. Install the plugin:
   ```bash
   uv add --dev plugin-name
   ```

2. Add it to `mkdocs.yml`:
   ```yaml
   plugins:
     - search
     - plugin-name
   ```

## 🚨 Troubleshooting

### Common Issues

1. **Build fails with plugin errors**
   - Make sure all required plugins are installed
   - Check the plugin configuration in `mkdocs.yml`

2. **GitHub Pages not updating**
   - Check the Actions tab for build status
   - Ensure the workflow completed successfully
   - Wait a few minutes for changes to propagate

3. **Local server won't start**
   - Check if port 8000 is already in use
   - Try a different port: `uv run mkdocs serve --dev-addr=127.0.0.1:8001`

4. **Images not displaying**
   - Ensure images are in the `docs/` directory
   - Use relative paths: `../docs/images/image.png`

### Getting Help

- Check the [MkDocs documentation](https://www.mkdocs.org/)
- Review the [Material theme documentation](https://squidfunk.github.io/mkdocs-material/)
- Check GitHub Actions logs for detailed error messages

## 📈 Analytics (Optional)

To add Google Analytics:

1. Get your tracking ID from Google Analytics
2. Uncomment and configure in `mkdocs.yml`:

```yaml
google_analytics:
  - G-XXXXXXXXXX
  - auto
```

## 🔄 Continuous Deployment

The GitHub Actions workflow automatically:

- Builds documentation on every push to `main`
- Deploys to GitHub Pages
- Handles pull requests for testing
- Provides build status and logs

## 📝 Best Practices

1. **Keep documentation up to date** with code changes
2. **Use descriptive commit messages** for documentation updates
3. **Test locally** before pushing changes
4. **Use relative links** for internal documentation references
5. **Include code examples** in your documentation
6. **Add alt text** to images for accessibility

## 🎉 Success!

Once everything is set up, your documentation will be:

- ✅ Automatically built and deployed
- ✅ Available at `https://nima-siboni.github.io/collectivecrossing/`
- ✅ Searchable and well-organized
- ✅ Mobile-responsive
- ✅ Easy to maintain and update

Happy documenting! 📚✨
