#!/usr/bin/env python3
"""
Comprehensive download solutions for Chinese users experiencing slow dataset downloads.
Provides multiple options and fallback mechanisms.
"""

import os
import sys
import time
import requests
import hashlib
from pathlib import Path
import zipfile
import tarfile
import gzip
import shutil

def check_file_integrity(filepath, expected_size=None, expected_checksum=None):
    """Check if downloaded file is valid."""
    if not os.path.exists(filepath):
        return False, "File does not exist"
    
    # Check file size
    actual_size = os.path.getsize(filepath)
    if expected_size and actual_size != expected_size:
        return False, f"Size mismatch: expected {expected_size}, got {actual_size}"
    
    # Check checksum if provided
    if expected_checksum:
        with open(filepath, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        if file_hash != expected_checksum:
            return False, f"Checksum mismatch: expected {expected_checksum}, got {file_hash}"
    
    return True, "File is valid"

def download_with_resume(url, filepath, chunk_size=8192, timeout=30):
    """Download file with resume capability and progress tracking."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Check if file already exists
        resume_pos = 0
        if os.path.exists(filepath):
            resume_pos = os.path.getsize(filepath)
            print(f"📁 Resuming download from {resume_pos} bytes")
        
        # Set up session with retries
        session = requests.Session()
        headers = {}
        if resume_pos > 0:
            headers['Range'] = f'bytes={resume_pos}-'
        
        # Get file info
        response = session.head(url, timeout=timeout)
        total_size = int(response.headers.get('content-length', 0))
        if resume_pos > 0:
            total_size += resume_pos
        
        # Download with progress
        mode = 'ab' if resume_pos > 0 else 'wb'
        with open(filepath, mode) as f:
            response = session.get(url, stream=True, headers=headers, timeout=timeout)
            response.raise_for_status()
            
            downloaded = resume_pos
            start_time = time.time()
            last_update = start_time
            
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Update progress every 2 seconds
                    current_time = time.time()
                    if current_time - last_update > 2:
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            speed = downloaded / (current_time - start_time) / (1024 * 1024)
                            print(f"📥 Progress: {progress:.1f}% ({downloaded/(1024*1024):.1f}MB) - {speed:.1f}MB/s")
                        else:
                            speed = downloaded / (current_time - start_time) / (1024 * 1024)
                            print(f"📥 Downloaded: {downloaded/(1024*1024):.1f}MB - {speed:.1f}MB/s")
                        last_update = current_time
        
        print(f"✅ Download completed: {downloaded/(1024*1024):.1f}MB")
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def get_alternative_sources():
    """Get alternative download sources for datasets."""
    return {
        'cifar10': {
            'filename': 'cifar-10-python.tar.gz',
            'expected_size': 170498071,
            'expected_checksum': '6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce',
            'sources': [
                {
                    'name': 'PyTorch Official',
                    'url': 'https://download.pytorch.org/datasets/cifar-10-python.tar.gz',
                    'priority': 1
                },
                {
                    'name': 'HuggingFace',
                    'url': 'https://huggingface.co/datasets/cifar10/resolve/main/cifar-10-python.tar.gz',
                    'priority': 2
                },
                {
                    'name': 'Original Source',
                    'url': 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
                    'priority': 3
                }
            ]
        },
        'cifar100': {
            'filename': 'cifar-100-python.tar.gz',
            'expected_size': 169001437,
            'expected_checksum': '85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7',
            'sources': [
                {
                    'name': 'PyTorch Official',
                    'url': 'https://download.pytorch.org/datasets/cifar-100-python.tar.gz',
                    'priority': 1
                },
                {
                    'name': 'HuggingFace',
                    'url': 'https://huggingface.co/datasets/cifar100/resolve/main/cifar-100-python.tar.gz',
                    'priority': 2
                },
                {
                    'name': 'Original Source',
                    'url': 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz',
                    'priority': 3
                }
            ]
        }
    }

def download_dataset(dataset_name, data_dir="./data"):
    """Download dataset using multiple sources with fallback."""
    sources = get_alternative_sources()
    
    if dataset_name not in sources:
        print(f"❌ Unknown dataset: {dataset_name}")
        return False
    
    config = sources[dataset_name]
    filename = config['filename']
    filepath = os.path.join(data_dir, filename)
    
    # Check if already downloaded and valid
    is_valid, message = check_file_integrity(
        filepath, 
        config['expected_size'], 
        config['expected_checksum']
    )
    
    if is_valid:
        print(f"✅ {dataset_name} already downloaded and valid")
        return True
    
    print(f"🚀 Downloading {dataset_name} from multiple sources...")
    
    # Try each source in priority order
    for source in sorted(config['sources'], key=lambda x: x['priority']):
        print(f"\n🌐 Trying {source['name']}: {source['url']}")
        
        if download_with_resume(source['url'], filepath):
            # Verify downloaded file
            is_valid, message = check_file_integrity(
                filepath, 
                config['expected_size'], 
                config['expected_checksum']
            )
            
            if is_valid:
                print(f"✅ {dataset_name} downloaded successfully from {source['name']}")
                return True
            else:
                print(f"⚠️ Downloaded file is invalid: {message}")
                # Remove invalid file
                if os.path.exists(filepath):
                    os.remove(filepath)
        else:
            print(f"❌ Failed to download from {source['name']}")
    
    print(f"❌ Failed to download {dataset_name} from all sources")
    return False

def extract_dataset(dataset_name, data_dir="./data"):
    """Extract downloaded dataset."""
    sources = get_alternative_sources()
    
    if dataset_name not in sources:
        print(f"❌ Unknown dataset: {dataset_name}")
        return False
    
    filename = sources[dataset_name]['filename']
    filepath = os.path.join(data_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"❌ Dataset file not found: {filepath}")
        return False
    
    print(f"📦 Extracting {dataset_name}...")
    
    try:
        if filename.endswith('.tar.gz'):
            with tarfile.open(filepath, 'r:gz') as tar:
                tar.extractall(data_dir)
            print(f"✅ {dataset_name} extracted successfully")
            return True
        else:
            print(f"❌ Unknown file format: {filename}")
            return False
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return False

def provide_manual_download_instructions():
    """Provide manual download instructions for users."""
    print("\n" + "=" * 60)
    print("📋 Manual Download Instructions for Chinese Users")
    print("=" * 60)
    
    print("\n🔧 If automatic downloads are too slow, you can manually download:")
    
    sources = get_alternative_sources()
    for dataset_name, config in sources.items():
        print(f"\n📁 {dataset_name.upper()}:")
        print(f"   Expected file: {config['filename']}")
        print(f"   Expected size: {config['expected_size'] / (1024*1024):.1f}MB")
        print(f"   Expected SHA256: {config['expected_checksum']}")
        
        for source in config['sources']:
            print(f"   🌐 {source['name']}: {source['url']}")
        
        print(f"   📂 Place the file in: ./data/{config['filename']}")
    
    print("\n💡 Tips for faster downloads in China:")
    print("   1. Use a VPN or proxy service")
    print("   2. Try downloading during off-peak hours")
    print("   3. Use download managers with resume capability")
    print("   4. Consider using cloud storage services")
    print("   5. Check if your institution has local mirrors")

def main():
    """Main function with multiple download options."""
    print("=" * 60)
    print("🇨🇳 Chinese User Dataset Download Solutions")
    print("=" * 60)
    
    # Create data directory
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Try automatic downloads
    print("\n🚀 Attempting automatic downloads...")
    
    success_cifar10 = download_dataset('cifar10', data_dir)
    if success_cifar10:
        extract_dataset('cifar10', data_dir)
    
    success_cifar100 = download_dataset('cifar100', data_dir)
    if success_cifar100:
        extract_dataset('cifar100', data_dir)
    
    # Provide manual instructions if automatic downloads fail
    if not (success_cifar10 and success_cifar100):
        provide_manual_download_instructions()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Download Summary:")
    print(f"   CIFAR-10: {'✅ Success' if success_cifar10 else '❌ Failed'}")
    print(f"   CIFAR-100: {'✅ Success' if success_cifar100 else '❌ Failed'}")
    print("=" * 60)

if __name__ == "__main__":
    main() 