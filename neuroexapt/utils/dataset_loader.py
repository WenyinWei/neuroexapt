"""
Advanced Dataset Loader with P2P Acceleration and Caching.

This module provides robust dataset downloading capabilities optimized for Chinese users,
including P2P acceleration, intelligent caching, and multiple mirror support.
"""

import os
import hashlib
import json
import time
import threading
import requests
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetMirror:
    """Represents a dataset mirror with metadata."""
    
    def __init__(self, name: str, base_url: str, region: str, priority: int = 1):
        self.name = name
        self.base_url = base_url.rstrip('/')
        self.region = region
        self.priority = priority
        self.last_check = 0
        self.response_time = float('inf')
        self.is_available = True
    
    def get_download_url(self, dataset_name: str, filename: str) -> str:
        """Generate download URL for a specific file."""
        return f"{self.base_url}/{dataset_name}/{filename}"
    
    def test_connectivity(self, timeout: int = 5) -> bool:
        """Test mirror connectivity and update response time."""
        try:
            start_time = time.time()
            response = requests.head(self.base_url, timeout=timeout)
            self.response_time = time.time() - start_time
            self.is_available = response.status_code == 200
            self.last_check = time.time()
            return self.is_available
        except Exception as e:
            logger.warning(f"Mirror {self.name} connectivity test failed: {e}")
            self.is_available = False
            self.response_time = float('inf')
            return False


class P2PDownloader:
    """P2P-accelerated downloader with chunked downloads and resume capability."""
    
    def __init__(self, chunk_size: int = 1024 * 1024, max_workers: int = 4):
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'NeuroExapt-Dataset-Loader/1.0'
        })
    
    def download_file(self, url: str, filepath: Path, 
                     resume: bool = True, progress_bar: bool = True) -> bool:
        """Download file with resume capability and progress tracking."""
        try:
            # Check if file exists and get its size
            existing_size = filepath.stat().st_size if filepath.exists() else 0
            
            # Get file info from server
            headers = {}
            if resume and existing_size > 0:
                headers['Range'] = f'bytes={existing_size}-'
            
            response = self.session.head(url, headers=headers, timeout=30)
            
            if response.status_code not in [200, 206]:
                logger.error(f"Failed to access {url}: {response.status_code}")
                return False
            
            total_size = int(response.headers.get('content-length', 0))
            if resume and existing_size > 0:
                total_size += existing_size
            
            # Open file for writing
            mode = 'ab' if resume and existing_size > 0 else 'wb'
            with open(filepath, mode) as f:
                if progress_bar:
                    pbar = tqdm(
                        total=total_size,
                        initial=existing_size,
                        unit='B',
                        unit_scale=True,
                        desc=f"Downloading {filepath.name}"
                    )
                
                # Download in chunks
                headers = {}
                if resume and existing_size > 0:
                    headers['Range'] = f'bytes={existing_size}-'
                
                response = self.session.get(url, headers=headers, stream=True, timeout=30)
                
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    if chunk:
                        f.write(chunk)
                        if progress_bar:
                            pbar.update(len(chunk))
                
                if progress_bar:
                    pbar.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Download failed for {url}: {e}")
            return False


class DatasetCache:
    """Intelligent dataset caching system."""
    
    def __init__(self, cache_dir: Union[str, Path] = "./data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load cache metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
        return {}
    
    def _save_metadata(self):
        """Save cache metadata."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache metadata: {e}")
    
    def get_cache_path(self, dataset_name: str, filename: str) -> Path:
        """Get cache path for a dataset file."""
        return self.cache_dir / dataset_name / filename
    
    def is_cached(self, dataset_name: str, filename: str, 
                  expected_size: Optional[int] = None) -> bool:
        """Check if file is cached and valid."""
        cache_path = self.get_cache_path(dataset_name, filename)
        
        if not cache_path.exists():
            return False
        
        # Check file size if expected size is provided
        if expected_size is not None:
            actual_size = cache_path.stat().st_size
            if actual_size != expected_size:
                logger.warning(f"Cache file size mismatch for {filename}")
                return False
        
        # Check metadata
        cache_key = f"{dataset_name}/{filename}"
        if cache_key in self.metadata:
            cache_info = self.metadata[cache_key]
            if 'checksum' in cache_info:
                # Verify checksum
                actual_checksum = self._calculate_checksum(cache_path)
                if actual_checksum != cache_info['checksum']:
                    logger.warning(f"Cache checksum mismatch for {filename}")
                    return False
        
        return True
    
    def _calculate_checksum(self, filepath: Path) -> str:
        """Calculate SHA256 checksum of file."""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def cache_file(self, dataset_name: str, filename: str, 
                   source_path: Path, expected_size: Optional[int] = None):
        """Cache a file with metadata."""
        cache_path = self.get_cache_path(dataset_name, filename)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file to cache
        import shutil
        shutil.copy2(source_path, cache_path)
        
        # Update metadata
        cache_key = f"{dataset_name}/{filename}"
        self.metadata[cache_key] = {
            'cached_at': time.time(),
            'size': cache_path.stat().st_size,
            'checksum': self._calculate_checksum(cache_path)
        }
        
        if expected_size is not None:
            self.metadata[cache_key]['expected_size'] = expected_size
        
        self._save_metadata()
        logger.info(f"Cached {filename} successfully")


class AdvancedDatasetLoader:
    """Advanced dataset loader with P2P acceleration, intelligent caching, and 迅雷 integration."""
    
    def __init__(self, cache_dir: str = "./data_cache", 
                 download_dir: str = "./data",
                 use_p2p: bool = True,
                 use_xunlei: bool = True,
                 max_retries: int = 3):
        self.cache = DatasetCache(cache_dir)
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.p2p_downloader = P2PDownloader() if use_p2p else None
        
        # Initialize 迅雷 downloader if requested
        self.xunlei_downloader = None
        if use_xunlei:
            try:
                from .xunlei_downloader import XunleiDownloader
                self.xunlei_downloader = XunleiDownloader()
                if self.xunlei_downloader.is_available:
                    logger.info("✅ 迅雷 integration enabled")
                else:
                    logger.info("⚠️ 迅雷 not available, falling back to standard download")
                    self.xunlei_downloader = None
            except ImportError:
                logger.info("⚠️ 迅雷 module not available, using standard download")
        
        self.max_retries = max_retries
        
        # Initialize mirrors with priority for Chinese users
        self.mirrors = self._initialize_mirrors()
        
        # Dataset configurations
        self.dataset_configs = {
            'cifar10': {
                'files': [
                    'cifar-10-python.tar.gz'
                ],
                'expected_size': 170498071,
                'checksum': '6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce',
                'direct_urls': [
                    'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
                ],
                'chinese_mirrors': [],
                'alternative_mirrors': [
                    'https://mirrors.tuna.tsinghua.edu.cn/pytorch-datasets/cifar-10-python.tar.gz',
                    'https://mirrors.ustc.edu.cn/pytorch-datasets/cifar-10-python.tar.gz',
                    'https://download.pytorch.org/datasets/cifar-10-python.tar.gz',
                    'https://huggingface.co/datasets/cifar10/resolve/main/cifar-10-python.tar.gz',
                ]
            },
            'cifar100': {
                'files': [
                    'cifar-100-python.tar.gz'
                ],
                'expected_size': 169001437,
                'checksum': '85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7',
                'direct_urls': [
                    'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
                ],
                'chinese_mirrors': [],
                'alternative_mirrors': [
                    'https://mirrors.tuna.tsinghua.edu.cn/pytorch-datasets/cifar-100-python.tar.gz',
                    'https://mirrors.ustc.edu.cn/pytorch-datasets/cifar-100-python.tar.gz',
                    'https://download.pytorch.org/datasets/cifar-100-python.tar.gz',
                    'https://huggingface.co/datasets/cifar100/resolve/main/cifar-100-python.tar.gz',
                ]
            },
            'mnist': {
                'files': [
                    'train-images-idx3-ubyte.gz',
                    'train-labels-idx1-ubyte.gz',
                    't10k-images-idx3-ubyte.gz',
                    't10k-labels-idx1-ubyte.gz'
                ],
                'expected_sizes': [9912422, 28881, 1648877, 4542],
                'direct_urls': [
                    'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
                    'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
                    'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
                    'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
                ],
                'chinese_mirrors': [
                    'https://download.pytorch.org/datasets/mnist/train-images-idx3-ubyte.gz',
                    'https://download.pytorch.org/datasets/mnist/train-labels-idx1-ubyte.gz',
                    'https://download.pytorch.org/datasets/mnist/t10k-images-idx3-ubyte.gz',
                    'https://download.pytorch.org/datasets/mnist/t10k-labels-idx1-ubyte.gz'
                ]
            }
        }
    
    def _initialize_mirrors(self) -> List[DatasetMirror]:
        """Initialize dataset mirrors with priority for Chinese users."""
        mirrors = [
            # Chinese mirrors (highest priority for Chinese users)
            DatasetMirror("Tsinghua", "https://mirrors.tuna.tsinghua.edu.cn/pytorch-datasets", "CN", 1),
            DatasetMirror("USTC", "https://mirrors.ustc.edu.cn/pytorch-datasets", "CN", 1),
            DatasetMirror("Huawei", "https://mirrors.huaweicloud.com/pytorch-datasets", "CN", 1),
            DatasetMirror("Aliyun", "https://mirrors.aliyun.com/pytorch-datasets", "CN", 2),
            DatasetMirror("Tencent", "https://mirrors.cloud.tencent.com/pytorch-datasets", "CN", 2),
            
            # International mirrors (fallback)
            DatasetMirror("PyTorch", "https://download.pytorch.org/datasets", "US", 3),
            
            # Direct dataset sources (lowest priority - slow for Chinese users)
            DatasetMirror("CIFAR-Direct", "https://www.cs.toronto.edu/~kriz", "CA", 4),
            DatasetMirror("MNIST-Direct", "http://yann.lecun.com/exdb/mnist", "US", 4),
        ]
        
        # Test mirrors in parallel
        threads = []
        for mirror in mirrors:
            thread = threading.Thread(target=mirror.test_connectivity)
            thread.start()
            threads.append(thread)
        
        for thread in threads:
            thread.join()
        
        # Sort by priority and availability
        mirrors.sort(key=lambda m: (m.priority, m.response_time))
        return mirrors
    
    def _get_best_mirror(self) -> Optional[DatasetMirror]:
        """Get the best available mirror."""
        for mirror in self.mirrors:
            if mirror.is_available:
                return mirror
        return None
    
    def download_dataset(self, dataset_name: str, force_download: bool = False) -> bool:
        """Download dataset with intelligent caching and P2P acceleration."""
        if dataset_name not in self.dataset_configs:
            logger.error(f"Unknown dataset: {dataset_name}")
            return False
        
        config = self.dataset_configs[dataset_name]
        dataset_dir = self.download_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if already downloaded
        if not force_download and self._is_dataset_complete(dataset_name, dataset_dir):
            logger.info(f"Dataset {dataset_name} already exists and is complete")
            return True
        
        # Download files with mirror fallback
        success = True
        for i, filename in enumerate(config['files']):
            cache_path = self.cache.get_cache_path(dataset_name, filename)
            target_path = dataset_dir / filename
            
            # Check cache first
            expected_size = config.get('expected_sizes', [config.get('expected_size')])[i] if 'expected_sizes' in config else config.get('expected_size')
            
            if self.cache.is_cached(dataset_name, filename, expected_size):
                logger.info(f"Using cached file: {filename}")
                if not target_path.exists():
                    import shutil
                    shutil.copy2(cache_path, target_path)
                continue
            
            # Try direct URLs first (Toronto University)
            download_success = False
            
            if 'direct_urls' in config and i < len(config['direct_urls']):
                direct_url = config['direct_urls'][i]
                logger.info(f"Trying direct source: {direct_url}")
                
                for attempt in range(self.max_retries):
                    if self._download_file_with_fallback(direct_url, target_path, filename):
                        # Verify file size to ensure it's not an HTML page
                        if target_path.exists() and target_path.stat().st_size > 10000000:  # > 10MB
                            download_success = True
                            break
                        else:
                            logger.warning(f"Downloaded file too small, likely HTML page")
                            target_path.unlink()  # Remove corrupted file
                    else:
                        logger.warning(f"Direct download attempt {attempt + 1} failed")
                        if attempt < self.max_retries - 1:
                            time.sleep(2 ** attempt)
            
            # Fallback to alternative mirrors if direct URLs fail
            if not download_success and 'alternative_mirrors' in config and i < len(config['alternative_mirrors']):
                for mirror_url in config['alternative_mirrors']:
                    logger.info(f"Trying alternative mirror: {mirror_url}")
                    
                    for attempt in range(self.max_retries):
                        if self._download_file_with_fallback(mirror_url, target_path, filename):
                            # Verify file size to ensure it's not an HTML page
                            if target_path.exists() and target_path.stat().st_size > 10000000:  # > 10MB
                                download_success = True
                                break
                            else:
                                logger.warning(f"Downloaded file too small, likely HTML page")
                                target_path.unlink()  # Remove corrupted file
                        else:
                            logger.warning(f"Alternative mirror download attempt {attempt + 1} failed")
                            if attempt < self.max_retries - 1:
                                time.sleep(2 ** attempt)
                    
                    if download_success:
                        break
            
            if download_success:
                # Cache the downloaded file
                self.cache.cache_file(dataset_name, filename, target_path, expected_size)
            else:
                logger.error(f"Failed to download {filename} from all sources")
                success = False
        
        # Extract dataset if download was successful
        if success:
            logger.info(f"Extracting {dataset_name} dataset...")
            if not self._extract_dataset(dataset_name, dataset_dir):
                logger.error(f"Failed to extract {dataset_name} dataset")
                success = False
        
        return success
    
    def _extract_dataset(self, dataset_name: str, dataset_dir: Path) -> bool:
        """Extract downloaded dataset files."""
        try:
            config = self.dataset_configs[dataset_name]
            
            for filename in config['files']:
                file_path = dataset_dir / filename
                
                if not file_path.exists():
                    continue
                
                if filename.endswith('.tar.gz'):
                    import tarfile
                    logger.info(f"Extracting {filename}...")
                    with tarfile.open(file_path, 'r:gz') as tar:
                        tar.extractall(dataset_dir)
                    logger.info(f"Extracted {filename} successfully")
                
                elif filename.endswith('.gz'):
                    import gzip
                    logger.info(f"Extracting {filename}...")
                    with gzip.open(file_path, 'rb') as f_in:
                        content = f_in.read()
                        with open(file_path.with_suffix(''), 'wb') as f_out:
                            f_out.write(content)
                    logger.info(f"Extracted {filename} successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return False
    
    def _download_file_with_fallback(self, url: str, filepath: Path, filename: str) -> bool:
        """Download file with robust fallback, 迅雷 integration, and chunked downloads."""
        # Try 迅雷 first (for Chinese users)
        if self.xunlei_downloader and self.xunlei_downloader.is_available:
            logger.info(f"🚀 Trying 迅雷 download: {filename}")
            if self.xunlei_downloader.download_with_xunlei(url, str(filepath.parent), filename):
                logger.info(f"✅ 迅雷 download started for {filename}")
                # Wait for download to complete
                logger.info(f"⏳ Waiting for 迅雷 download to complete...")
                max_wait_time = 300  # 5 minutes
                wait_interval = 5    # Check every 5 seconds
                waited_time = 0
                
                while waited_time < max_wait_time:
                    if filepath.exists() and filepath.stat().st_size > 10000000:  # > 10MB
                        logger.info(f"✅ 迅雷 download completed: {filename}")
                        return True
                    time.sleep(wait_interval)
                    waited_time += wait_interval
                    logger.info(f"⏳ Still waiting... ({waited_time}s/{max_wait_time}s)")
                
                logger.warning(f"⚠️ 迅雷 download timeout for {filename}, trying alternatives")
            else:
                logger.warning(f"⚠️ 迅雷 download failed for {filename}, trying alternatives")
        
        # Try P2P downloader
        if self.p2p_downloader:
            if self.p2p_downloader.download_file(url, filepath):
                return True
        
        # Fallback to robust chunked download
        return self._download_with_chunks(url, filepath, filename)
    
    def _download_with_chunks(self, url: str, filepath: Path, filename: str) -> bool:
        """
        Download file with chunked transfer and robust error handling.
        
        Args:
            url (str): Download URL
            filepath (Path): Local file path
            filename (str): Filename for logging
            
        Returns:
            bool: True if download successful
        """
        try:
            # Use session for connection pooling and retries
            session = requests.Session()
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry
            
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            
            # Head request to get file size
            try:
                head_response = session.head(url, timeout=10, allow_redirects=True)
                total_size = int(head_response.headers.get('content-length', 0))
            except:
                total_size = 0
            
            # Download with progress tracking
            response = session.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            downloaded_size = 0
            chunk_size = 8192  # 8KB chunks
            last_progress_time = time.time()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:  # chunk should already be bytes from iter_content
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        # Progress update every 5 seconds or 1MB
                        current_time = time.time()
                        if (current_time - last_progress_time > 5 or 
                            downloaded_size % (1024 * 1024) == 0):
                            
                            if total_size > 0:
                                progress = (downloaded_size / total_size) * 100
                                speed = downloaded_size / (current_time - last_progress_time + 1) / (1024 * 1024)
                                logger.info(f"📥 {filename}: {progress:.1f}% ({downloaded_size / (1024*1024):.1f}MB) - {speed:.1f}MB/s")
                            else:
                                speed = downloaded_size / (current_time - last_progress_time + 1) / (1024 * 1024)
                                logger.info(f"📥 {filename}: {downloaded_size / (1024*1024):.1f}MB - {speed:.1f}MB/s")
                            
                            last_progress_time = current_time
            
            # Final progress
            if total_size > 0:
                progress = (downloaded_size / total_size) * 100
                logger.info(f"✅ {filename}: Download complete ({progress:.1f}%)")
            else:
                logger.info(f"✅ {filename}: Download complete ({downloaded_size / (1024*1024):.1f}MB)")
            
            return True
            
        except requests.exceptions.Timeout:
            logger.error(f"⏰ {filename}: Download timeout")
            if filepath.exists():
                filepath.unlink()  # Remove partial file
            return False
        except requests.exceptions.ConnectionError:
            logger.error(f"🔌 {filename}: Connection error")
            if filepath.exists():
                filepath.unlink()  # Remove partial file
            return False
        except requests.exceptions.HTTPError as e:
            logger.error(f"🌐 {filename}: HTTP error {e.response.status_code}")
            if filepath.exists():
                filepath.unlink()  # Remove partial file
            return False
        except Exception as e:
            logger.error(f"❌ {filename}: Unexpected error: {str(e)}")
            if filepath.exists():
                filepath.unlink()  # Remove partial file
            return False
    
    def _is_dataset_complete(self, dataset_name: str, dataset_dir: Path) -> bool:
        """Check if dataset is complete."""
        config = self.dataset_configs[dataset_name]
        
        for filename in config['files']:
            filepath = dataset_dir / filename
            if not filepath.exists():
                return False
            
            # Check file size
            expected_size = config.get('expected_sizes', [config.get('expected_size')])[config['files'].index(filename)] if 'expected_sizes' in config else config.get('expected_size')
            if expected_size and filepath.stat().st_size != expected_size:
                return False
        
        return True
    
    def get_cifar10_dataloaders(self, batch_size: int = 128, 
                               num_workers: int = 2,
                               download: bool = True,
                               force_download: bool = False) -> Tuple[DataLoader, DataLoader]:
        """Get CIFAR-10 data loaders with advanced downloading."""
        if download:
            if not self.download_dataset('cifar10', force_download):
                raise RuntimeError("Failed to download CIFAR-10 dataset")
        
        # Data transformations
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # Datasets
        trainset = torchvision.datasets.CIFAR10(
            root=str(self.download_dir), train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR10(
            root=str(self.download_dir), train=False, download=True, transform=transform_test
        )

        # Data loaders
        train_loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        test_loader = DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        return train_loader, test_loader
    
    def get_cifar100_dataloaders(self, batch_size: int = 128,
                                num_workers: int = 2,
                                download: bool = True,
                                force_download: bool = False) -> Tuple[DataLoader, DataLoader]:
        """Get CIFAR-100 data loaders with advanced downloading."""
        if download:
            if not self.download_dataset('cifar100', force_download):
                raise RuntimeError("Failed to download CIFAR-100 dataset")
        
        # Data transformations
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        # Datasets
        trainset = torchvision.datasets.CIFAR100(
            root=str(self.download_dir), train=True, download=False, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR100(
            root=str(self.download_dir), train=False, download=False, transform=transform_test
        )

        # Data loaders
        train_loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        test_loader = DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        return train_loader, test_loader
    
    def clear_cache(self, dataset_name: Optional[str] = None):
        """Clear cache for specific dataset or all datasets."""
        if dataset_name:
            cache_path = self.cache.cache_dir / dataset_name
            if cache_path.exists():
                import shutil
                shutil.rmtree(cache_path)
                logger.info(f"Cleared cache for {dataset_name}")
        else:
            import shutil
            shutil.rmtree(self.cache.cache_dir)
            self.cache.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cleared all cache")
    
    def get_cache_info(self) -> Dict:
        """Get cache information."""
        info = {
            'cache_dir': str(self.cache.cache_dir),
            'total_size': 0,
            'datasets': {}
        }
        
        for dataset_dir in self.cache.cache_dir.iterdir():
            if dataset_dir.is_dir():
                dataset_size = sum(f.stat().st_size for f in dataset_dir.rglob('*') if f.is_file())
                info['datasets'][dataset_dir.name] = {
                    'size': dataset_size,
                    'files': len(list(dataset_dir.rglob('*')))
                }
                info['total_size'] += dataset_size
        
        return info


# Convenience function for backward compatibility
def get_cifar10_dataloaders(batch_size: int = 128, num_workers: int = 2,
                           cache_dir: str = "./data_cache",
                           download_dir: str = "./data",
                           use_p2p: bool = True) -> Tuple[DataLoader, DataLoader]:
    """Get CIFAR-10 data loaders with advanced downloading capabilities."""
    loader = AdvancedDatasetLoader(cache_dir=cache_dir, download_dir=download_dir, use_p2p=use_p2p)
    return loader.get_cifar10_dataloaders(batch_size, num_workers) 