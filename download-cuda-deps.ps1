$ErrorActionPreference = 'Stop'
$cache = Join-Path $PSScriptRoot 'docker\torch_wheels\cu128'
$files = @(
  @('nvidia-cublas-cu12','nvidia_cublas_cu12-12.8.4.1-py3-none-manylinux_2_27_x86_64.whl'),
  @('nvidia-cufft-cu12','nvidia_cufft_cu12-11.3.3.83-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl'),
  @('nvidia-curand-cu12','nvidia_curand_cu12-10.3.9.90-py3-none-manylinux_2_27_x86_64.whl'),
  @('nvidia-cusolver-cu12','nvidia_cusolver_cu12-11.7.3.90-py3-none-manylinux_2_27_x86_64.whl'),
  @('nvidia-cusparse-cu12','nvidia_cusparse_cu12-12.5.8.93-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl'),
  @('nvidia-cusparselt-cu12','nvidia_cusparselt_cu12-0.7.1-py3-none-manylinux2014_x86_64.whl'),
  @('nvidia-nccl-cu12','nvidia_nccl_cu12-2.27.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl'),
  @('nvidia-nvshmem-cu12','nvidia_nvshmem_cu12-3.3.20-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl'),
  @('nvidia-nvtx-cu12','nvidia_nvtx_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl'),
  @('nvidia-nvjitlink-cu12','nvidia_nvjitlink_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl'),
  @('nvidia-cufile-cu12','nvidia_cufile_cu12-1.13.1.3-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl')
)
foreach ($entry in $files) {
  $target = Join-Path $cache $entry[1]
  if (Test-Path $target) { Write-Output "EXISTS $($entry[1])"; continue }
  $url = "https://pypi.nvidia.cn/$($entry[0])/$($entry[1])"
  Write-Output "DOWNLOAD $url"
  curl.exe --fail --location --retry 3 --connect-timeout 30 --output $target $url
  Write-Output "DONE $($entry[1])"
}
