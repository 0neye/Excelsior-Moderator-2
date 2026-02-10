"""
ZipApp - Creates a zip archive containing all Git-tracked files, the database file, environment variables, the lightgbm model, and rating metadata.
Uses Git itself to determine which files to include, ensuring robustness and consistency with version control.
"""

import argparse
import subprocess
import zipfile
from pathlib import Path


def get_git_tracked_files(root_path):
    """
    Get all files tracked by Git using git ls-files command.
    
    Args:
        root_path: Path to the root directory of the Git repository
        
    Returns:
        List of Path objects for tracked files
    """
    try:
        # Run git ls-files to get all tracked files
        result = subprocess.run(
            ['git', 'ls-files'],
            cwd=root_path,
            capture_output=True,
            text=True,
            check=True
        )
        
        tracked_files = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():  # Skip empty lines
                file_path = root_path / line.strip()
                if file_path.exists():
                    tracked_files.append(file_path)
        
        return tracked_files
    except subprocess.CalledProcessError as e:
        print(f"Error running git ls-files: {e}")
        print("Make sure you're in a Git repository.")
        return []
    except FileNotFoundError:
        print("Git is not installed or not in PATH.")
        return []


def create_zip_archive(output_name='excelsior_bot.zip', code_only=False):
    """
    Create a zip archive containing all Git-tracked files, the database file, .env file, lightgbm model, and rating metadata.
    
    Args:
        output_name: Name of the output zip file
        code_only: If True, include only Git-tracked files (skip database, .env, model, rating metadata)
    """
    root_path = Path(__file__).parent.resolve()
    
    # Get all Git-tracked files
    print("Getting Git-tracked files...")
    tracked_files = get_git_tracked_files(root_path)
    
    if not tracked_files:
        print("No tracked files found. Make sure you're in a Git repository.")
        return
    
    # Files to include
    files_to_include = list(tracked_files)
    
    # Add data files unless --code-only
    if not code_only:
        # Add database file if it exists
        database_file = root_path / 'excelsior.db'
        if database_file.exists():
            if database_file not in files_to_include:
                files_to_include.append(database_file)
                print(f"Added database file: {database_file.name}")
        else:
            print(f"Warning: Database file {database_file.name} not found.")
        
        # Add .env file if it exists
        env_file = root_path / '.env'
        if env_file.exists():
            if env_file not in files_to_include:
                files_to_include.append(env_file)
                print(f"Added environment file: {env_file.name}")
        else:
            print(f"Warning: Environment file {env_file.name} not found.")
        
        # Add lightgbm model file if it exists
        model_file = root_path / 'models' / 'lightgbm_model.joblib'
        if model_file.exists():
            if model_file not in files_to_include:
                files_to_include.append(model_file)
                print(f"Added model file: {model_file.relative_to(root_path)}")
        else:
            print(f"Warning: Model file {model_file.relative_to(root_path)} not found.")
        
        # Add rating metadata file if it exists
        rating_metadata_file = root_path / 'data' / 'rating_metadata.json'
        if rating_metadata_file.exists():
            if rating_metadata_file not in files_to_include:
                files_to_include.append(rating_metadata_file)
                print(f"Added rating metadata file: {rating_metadata_file.relative_to(root_path)}")
        else:
            print(f"Warning: Rating metadata file {rating_metadata_file.relative_to(root_path)} not found.")
    else:
        print("Code-only mode: skipping database, .env, model, and rating metadata")
    
    # Skip the zip file itself if it exists
    zip_path = root_path / output_name
    files_to_include = [f for f in files_to_include if f != zip_path]
    
    if not files_to_include:
        print("No files found to include in the archive.")
        return
    
    # Create zip archive
    print(f"\nCreating zip archive: {output_name}")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in files_to_include:
            # Use relative path in zip archive
            try:
                arcname = file_path.relative_to(root_path)
                zipf.write(file_path, arcname)
                print(f"Added: {arcname}")
            except Exception as e:
                print(f"Error adding {file_path}: {e}")
    
    print(f"\nSuccessfully created {output_name}")
    print(f"Total files: {len(files_to_include)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a zip archive of the Excelsior Moderator project.')
    parser.add_argument('--code-only', action='store_true', help='Include only Git-tracked files (skip database, .env, model, rating metadata)')
    args = parser.parse_args()
    create_zip_archive(code_only=args.code_only)
