"""
Quick setup script to install dependencies and test the installation.
"""

import subprocess
import sys
import os


def install_requirements():
    """Install required packages."""
    print("Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True


def test_imports():
    """Test if all required packages can be imported."""
    print("\nTesting imports...")
    
    packages = [
        "torch",
        "numpy", 
        "transformers",
        "matplotlib",
        "tqdm"
    ]
    
    failed_imports = []
    
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {failed_imports}")
        return False
    else:
        print("\n✅ All packages imported successfully!")
        return True


def create_directories():
    """Create necessary directories."""
    print("\nCreating directories...")
    
    directories = ["data", "models", "logs"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created/verified directory: {directory}")


def main():
    """Main setup function."""
    print("=== Simple LLM Setup ===\n")
    
    # Check if we're in the right directory
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found. Please run this script from the project root directory.")
        return
    
    # Install requirements
    if not install_requirements():
        return
    
    # Test imports
    if not test_imports():
        print("\n❌ Setup failed. Please check the error messages above.")
        return
    
    # Create directories
    create_directories()
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the training example: python examples/train_simple_model.py")
    print("2. Try text generation: python examples/generate_text.py")
    print("3. Explore the Jupyter notebook: jupyter notebook notebooks/simple_llm_exploration.ipynb")


if __name__ == "__main__":
    main()
