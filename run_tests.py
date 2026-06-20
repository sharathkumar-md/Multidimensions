#!/usr/bin/env python3
import sys
import subprocess
from pathlib import Path

def main():
    root_dir = Path(__file__).parent.resolve()
    subprojects = ["01_ocr", "02_llm_eval", "03_rag"]
    
    print("=" * 60)
    print("Running MultiDimensions Test Suites")
    print("=" * 60)
    
    failed = False
    for subproject in subprojects:
        subproject_path = root_dir / subproject
        if not subproject_path.exists():
            print(f"\n[!] Subproject directory {subproject} does not exist. Skipping.")
            continue
            
        print(f"\n>>> Running tests in {subproject} ...")
        print("-" * 60)
        
        # Run pytest inside the subproject directory to isolate sys.path
        result = subprocess.run(
            [sys.executable, "-m", "pytest"],
            cwd=str(subproject_path),
        )
        
        if result.returncode != 0:
            print(f"\n[FAIL] Tests in {subproject} failed with exit code {result.returncode}")
            failed = True
        else:
            print(f"\n[PASS] Tests in {subproject} completed successfully.")
            
    print("\n" + "=" * 60)
    if failed:
        print("Overall Status: FAILED (one or more test suites failed)")
        print("=" * 60)
        sys.exit(1)
    else:
        print("Overall Status: ALL PASSED")
        print("=" * 60)
        sys.exit(0)

if __name__ == "__main__":
    main()
