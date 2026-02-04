"""
SA-1B Dataset Quality Analysis with CleanVision

This script randomly selects tar files from SA-1B dataset,
extracts images, and analyzes them using CleanVision.
"""

import os
import random
import shutil
import tarfile
import glob
from pathlib import Path

# Try to import cleanvision, provide installation hint if not found
try:
    from cleanvision import Imagelab
except ImportError:
    print("CleanVision not installed. Please install it with:")
    print("pip install cleanvision")
    exit(1)


# ============== Configuration ==============
SA1B_PATH = os.path.expanduser("~/nfs-australia-east-1/NekoDiff/SA-1B")
NUM_TAR_FILES = 5
WORK_DIR = Path(__file__).parent.resolve()
EXTRACTED_DIR = WORK_DIR / "sa1b_extracted_images"
REPORT_PATH = WORK_DIR / "cleanvision_report"
# ===========================================


def find_tar_files(dataset_path: str) -> list:
    """Find all tar files in the SA-1B dataset directory."""
    tar_pattern = os.path.join(dataset_path, "**/*.tar")
    tar_files = glob.glob(tar_pattern, recursive=True)
    
    if not tar_files:
        # Also try direct tar files without subdirectories
        tar_pattern = os.path.join(dataset_path, "*.tar")
        tar_files = glob.glob(tar_pattern)
    
    return tar_files


def select_random_tar_files(tar_files: list, num_files: int) -> list:
    """Randomly select tar files from the available list."""
    if len(tar_files) < num_files:
        print(f"Warning: Only {len(tar_files)} tar files found, using all of them.")
        return tar_files
    return random.sample(tar_files, num_files)


def extract_images_from_tar(tar_path: str, extract_dir: Path) -> int:
    """Extract image files from a tar archive."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}
    extracted_count = 0
    
    try:
        with tarfile.open(tar_path, 'r') as tar:
            for member in tar.getmembers():
                if member.isfile():
                    ext = os.path.splitext(member.name)[1].lower()
                    if ext in image_extensions:
                        # Extract to flat directory with unique name
                        member.name = os.path.basename(member.name)
                        # Avoid name collisions by prefixing with tar filename
                        tar_basename = os.path.basename(tar_path).replace('.tar', '')
                        new_name = f"{tar_basename}_{member.name}"
                        
                        # Extract to temp location then rename
                        tar.extract(member, extract_dir)
                        old_path = extract_dir / member.name
                        new_path = extract_dir / new_name
                        if old_path.exists() and old_path != new_path:
                            shutil.move(str(old_path), str(new_path))
                        
                        extracted_count += 1
    except Exception as e:
        print(f"Error extracting {tar_path}: {e}")
    
    return extracted_count


def run_cleanvision_analysis(image_dir: Path) -> Imagelab:
    """Run CleanVision analysis on extracted images."""
    print(f"\nRunning CleanVision analysis on: {image_dir}")
    print("-" * 50)
    
    imagelab = Imagelab(data_path=str(image_dir))
    
    # Find all issues
    imagelab.find_issues()
    
    return imagelab


def main():
    print("=" * 60)
    print("SA-1B Dataset Quality Analysis with CleanVision")
    print("=" * 60)
    
    # Check if SA-1B path exists
    if not os.path.exists(SA1B_PATH):
        print(f"\nError: SA-1B dataset path not found: {SA1B_PATH}")
        print("Please ensure the NFS mount is accessible or update SA1B_PATH.")
        return
    
    # Create extraction directory
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nExtraction directory: {EXTRACTED_DIR}")
    
    # Find tar files
    print(f"\nSearching for tar files in: {SA1B_PATH}")
    tar_files = find_tar_files(SA1B_PATH)
    print(f"Found {len(tar_files)} tar files")
    
    if not tar_files:
        print("Error: No tar files found in the dataset directory.")
        return
    
    # Randomly select tar files
    selected_tars = select_random_tar_files(tar_files, NUM_TAR_FILES)
    print(f"\nRandomly selected {len(selected_tars)} tar files:")
    for tar_path in selected_tars:
        print(f"  - {os.path.basename(tar_path)}")
    
    # Extract images from selected tar files
    print("\nExtracting images...")
    total_extracted = 0
    for tar_path in selected_tars:
        print(f"  Extracting: {os.path.basename(tar_path)}")
        count = extract_images_from_tar(tar_path, EXTRACTED_DIR)
        total_extracted += count
        print(f"    Extracted {count} images")
    
    print(f"\nTotal images extracted: {total_extracted}")
    
    if total_extracted == 0:
        print("Error: No images were extracted. Check tar file contents.")
        return
    
    # Run CleanVision analysis
    imagelab = run_cleanvision_analysis(EXTRACTED_DIR)
    
    # Print report to console
    print("\n" + "=" * 60)
    print("CleanVision Report")
    print("=" * 60)
    imagelab.report()
    
    # Save detailed report
    REPORT_PATH.mkdir(parents=True, exist_ok=True)
    imagelab.report(save_path=str(REPORT_PATH))
    print(f"\nDetailed report saved to: {REPORT_PATH}")
    
    # Get issue summary
    print("\n" + "=" * 60)
    print("Issue Summary")
    print("=" * 60)
    
    issues = imagelab.issue_summary
    print(issues)
    
    # Save issue summary to CSV
    summary_csv = WORK_DIR / "issue_summary.csv"
    issues.to_csv(summary_csv)
    print(f"\nIssue summary saved to: {summary_csv}")
    
    # Get images with issues
    issues_df = imagelab.issues
    issues_csv = WORK_DIR / "all_issues.csv"
    issues_df.to_csv(issues_csv)
    print(f"Detailed issues saved to: {issues_csv}")
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    
    # Cleanup prompt
    print(f"\nNote: Extracted images are in: {EXTRACTED_DIR}")
    print("You can delete this folder after reviewing the report.")


if __name__ == "__main__":
    main()
