# NeuroExapt Changelog

All notable changes to this project will be documented in this file.

## [Unreleased] - 2024-07-19

### Added
- 🧠 **Comprehensive Documentation System**
  - Automated Doxygen-based documentation generation
  - GitHub Actions workflow for auto-deployment to GitHub Pages
  - Smart markdown file integration and categorization
  - Modern responsive UI with dark mode support
  - Live code example execution and results

- 🔧 **Unified Logging System**
  - Centralized `logging_utils.py` module
  - Environment-based configuration (LOG_LEVEL, ENABLE_CONSOLE, LOG_FILE)
  - Logger instance caching for consistent state
  - Backward compatibility with existing DebugPrinter interface

### Fixed
- ❌ **Circular Import Resolution**
  - Eliminated circular dependency between `advanced_morphogenesis.py` and `enhanced_dnm_framework.py`
  - Moved shared logger functionality to separate module
  - Updated all inconsistent debug calls across codebase

- 🧹 **Project Structure Cleanup**
  - Moved test files from root to `test/legacy/`
  - Organized demo files in `examples/`
  - Moved utility scripts to `utilities/`
  - Cleaned up root directory with 60+ files reorganized

### Changed
- 📝 **Documentation Organization**
  - Detailed changelogs moved to `docs/changelogs/`
  - Root directory now contains concise project files
  - Automated categorization of 77+ markdown files

### Technical Details
For detailed technical information, see:
- [Documentation System Setup](docs/changelogs/DOCUMENTATION_SYSTEM_SETUP.md)
- [Project Structure Refactor](docs/changelogs/PROJECT_STRUCTURE_REFACTOR.md)
- [Logger Method Fixes](docs/changelogs/LOGGER_METHOD_FIXES_SUMMARY.md)

---

## Legend
- 🧠 Features
- 🔧 Improvements  
- ❌ Bug Fixes
- 🧹 Maintenance
- 📝 Documentation
- ⚡ Performance