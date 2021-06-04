"""Reconstruct some read data using tike multiprobe Odstrcil method."""

import os

import click
import h5py
#import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import skimage.restoration
import sys
import tike.ptycho
#import tike.view
import time
# import tifffile
import shutil
import cupy
import logging
import re
import pynvml as pm
import socket

def convert_positions_to_pixel(positions, probe_size, detector_distance,
                               detector_pixel_size, incident_wavelength):
    pos2det_const = (detector_pixel_size * probe_size) / (
        detector_distance * 1e-10 * incident_wavelength)
    return positions * pos2det_const.astype('float32')


def clip_data(scan, data, prb, fraction=1):

    # Add offset
    scan[..., 1] -= np.min(scan[..., 1])
    scan[..., 0] -= np.min(scan[..., 0])

    # Filter out of bounds positions
    in_bounds = np.logical_and(
        np.logical_and(
            scan[..., 1] >= 0,
            scan[..., 1] < np.inf,
        ),
        np.logical_and(
            scan[..., 0] >= 0,
            scan[..., 0] < np.inf,
        ),
    )

    # Add offset
    scan[..., 1] -= np.min(scan[..., 1])
    scan[..., 0] -= np.min(scan[..., 0])

    # Choose a random subset of positions
    nscan = scan.shape[-2]
    np.random.seed(0)
    random_subset = np.random.rand(nscan) <= fraction

    keep_me = np.logical_and(in_bounds, random_subset)

    scan = scan[None, keep_me, ...]

    logging.info(f'scan range is (0, {np.max(scan[..., 0])}), (0, {np.max(scan[..., 1])}).')
    logging.info(f'scan positions are {scan.shape}, {scan.dtype}')
    #plt.figure(dpi=600)
    #plt.plot(scan[..., 1], scan[..., 0], '.', color='blue')
    #plt.axis('equal')
    #plt.gca().invert_yaxis()
    #plt.savefig(f'{folder}/scan.png')

    data = data[None, keep_me, ...]

    logging.info(f'data is {data.shape}, {data.dtype}')
    #plt.figure(dpi=600)
    #plt.imshow(data[0, 0])
    #plt.savefig(f'{folder}/frame.png')
    #plt.close('all')

    logging.info(f'probe is {prb.shape}, {prb.dtype}')

    return (
        np.ascontiguousarray(scan),
        np.ascontiguousarray(data),
        np.ascontiguousarray(prb),
    )


def read_data_catalyst(ifile):
    #name = 'input/catalyst/extracted_scan201.h5'
    #name = '/gdata/RAVEN/bicer/2020-1/comm_33IDD/extracted_tekin/extracted_scan350.h5'
    #name = ifile #'/lus/grand/projects/hp-ptycho/bicer/ptycho/comm_33IDD/extracted_tekin/extracted_scan350.h5'
    # /grand/hp-ptycho/bicer/ptycho/comm_33IDD/globus_automate/input/300 + /extracted_scan300.h5
    iid = re.findall(r'\d+', ifile)
    name = f"{ifile}/extracted_scan{iid[-1]}.h5"
    logging.info("View file path: {name}")

    with h5py.File(name, 'r') as fid:
        scan = fid['/positions_0'][:] * 34068864.0
        scan = np.array(scan, dtype='float32', order='C')[np.newaxis]
        # scan = np.flip(scan, -1)

        data = np.array(fid['data'][:], order='C')[np.newaxis]
        data = np.fft.fftshift(data, axes=(-1, -2)).astype('float32')
        data = np.swapaxes(data, -1, -2)

        probe = fid['/recprobe'][:][np.newaxis, np.newaxis, np.newaxis]

    return (
        np.ascontiguousarray(scan),
        np.ascontiguousarray(data),
        np.ascontiguousarray(probe),
    )


def orthogonalize_gs(xp, x):
    """Gram-schmidt orthogonalization for complex arrays.

    x : (..., nmodes, :, :) array_like
        The array with modes in the -3 dimension.

    TODO: Possibly a faster implementation would use QR decomposition.
    """
    def inner(x, y, axis=None):
        """Return the complex inner product of x and y along axis."""
        return xp.sum(xp.conj(x) * y, axis=axis, keepdims=True)

    def norm(x, axis=None):
        """Return the complex vector norm of x along axis."""
        return xp.sqrt(inner(x, x, axis=axis))

    # Reshape x into a 2D array
    unflat_shape = x.shape
    nmodes = unflat_shape[-3]
    x_ortho = x.reshape(*unflat_shape[:-2], -1)

    for i in range(1, nmodes):
        u = x_ortho[..., 0:i, :]
        v = x_ortho[..., i:i + 1, :]
        projections = u * inner(u, v, axis=-1) / inner(u, u, axis=-1)
        x_ortho[..., i:i + 1, :] -= xp.sum(projections, axis=-2, keepdims=True)

    if __debug__:
        # Test each pair of vectors for orthogonality
        for i in range(nmodes):
            for j in range(i):
                error = abs(
                    inner(x_ortho[..., i:i + 1, :],
                          x_ortho[..., j:j + 1, :],
                          axis=-1))
                assert xp.all(error < 1e-5), (
                    f"Some vectors are not orthogonal!, {error}, {error.shape}"
                )

    return x_ortho.reshape(unflat_shape)


def orthogonalize_eig(xp, x):
    """Orthogonalize modes of x using a eigenvectors of the pairwise dot product.

    Parameters
    ----------
    x : (nmodes, probe_shape * probe_shape) array_like complex64
        An array of the probe modes vectorized
    """

    nmodes = 3
    # 'A' holds the dot product of all possible mode pairs
    A = np.empty((nmodes, nmodes), dtype='complex64')
    # TODO: Redundant computations because 'A' is symmetric?
    for i in range(nmodes):
        for j in range(nmodes):
            # Perhaps this should be an inner product
            # which means of the vectors should be conjugated?
            A[i, j] = np.sum(x[i] * x[j])

    values, vectors = np.linalg.eig(A)

    x_new = np.zeros_like(x)
    for i in range(nmodes):
        for j in range(nmodes):
            x_new[j] += vectors[i, j] * x[i]

    # Sort new modes by eigen value in decending order
    for order in xp.argsort(-values):
        x[order] = x_new[order]

    return x


def add_modes_random_phase(probe, nmodes):
    """Initialize probe modes by random phase shifts to the first mode.

    Parameters
    ----------
    probe : array
        A probe with at least one incoherent mode.
    nmodes : int
        The number of desired modes.
    """
    all_modes = np.empty((*probe.shape[:-3], nmodes, *probe.shape[-2:]),
                         dtype='complex64')
    pw = probe.shape[-1]
    for m in range(nmodes):
        if m < probe.shape[-3]:
            # copy existing mode
            all_modes[..., m, :, :] = probe[..., m, :, :]
        else:
            # randomly shift the first mode
            xshift = np.exp(-2j * np.pi * ((np.arange(0, pw) / pw + 1 /
                                            (pw * 2)) - 0.5) *
                            np.random.rand())
            yshift = np.exp(-2j * np.pi * ((np.arange(0, pw) / pw + 1 /
                                            (pw * 2)) - 0.5) *
                            np.random.rand())
            all_modes[..., m, :, :] = (probe[..., 0, :, :] * xshift[None] *
                                       yshift[:, None])
    return all_modes


def list_of_avail_gpus(mem_threshold):
  pm.nvmlInit()

  ngpus = pm.nvmlDeviceGetCount()
  ghandles = []
  for i in range(ngpus):
    handle = pm.nvmlDeviceGetHandleByIndex(i)
    ghandles.append(handle) 

  m_infos = []
  for handle in ghandles:
    #m_infos[0].total; m_infos[0].free; m_infos[0].used;
    m_infos.append(pm.nvmlDeviceGetMemoryInfo(handle)) 

  p_infos = {}
  for i in range(len(ghandles)):
    proc_infos = pm.nvmlDeviceGetComputeRunningProcesses(ghandles[i])
    p_infos[i] = {}
    for proc_info in proc_infos:
      #p_infos[gpud_id][pid]="process-name";
      p_infos[i][proc_info.pid] = pm.nvmlSystemGetProcessName(proc_info.pid)
 
  avail_gpus = []
  for i in range(len(ghandles)):
    p_info = p_infos[i]
    m_info = m_infos[i]
    if(len(p_info)==0 and # there is no process running
        (m_info.used//(mem_threshold * 2**20)) < mem_threshold): # the memory utilization is smaller than 50MiB
      avail_gpus.append(i)

  return avail_gpus


@click.command()
@click.option('--use-mpi', is_flag=True, 
              help='Use mpu for multi-process/node multi-gpu configurations.')
@click.option('--recover-probe', is_flag=True, 
              help='Recover probe.') 
@click.option('--recover-psi', is_flag=True,
              help='Recover object.') 
@click.option('--overwrite', is_flag=True, 
              help='Clears the output directory before writing data.')
@click.option('--ngpu', default=1, type=click.INT,
              help='Number of GPUs to use for reconstruction.')
@click.option('--gpu-id', default=None, type=click.INT,
              help='GPU id to pin this cupy. Currently, if this is set then ngpu has to be 1.')
@click.option('--auto-pin', is_flag=True,
              help="Automatically check GPUs for availability and pin to the free GPU. "
                   "There should be no active process on the GPU and the memory utilization "
                   "should be below gpu-mem-threshold to recognize a GPU as available.")
@click.option('--gpu-mem-threshold', default=50, type=click.INT,
              help="Memory constraint for the auto pin functionality. E.g., if this is "
                   "set to 50 (default), GPU memory utilization needs to be smaller than 50MiB.")
@click.option('--niter', default=100, type=click.INT,
              help='Number of iterations.')
@click.option('--output-freq', default=10, type=click.INT,
              help='Output frequency. Images are stored after every 10 iterations, if this is set to 10.')
@click.option('--model', default='gaussian', type=click.STRING,
              help='Noise model for cost function.')
@click.option('--ifile', type=click.STRING,
              help='Input file directory. Expects Tekin\'s single file format.')
@click.option('--algorithm', type=click.STRING, default='cgrad',
              help='Algorithm for reconstructions:(cgrad, lstsq_grad)')
@click.option('--nmodes', type=click.INT, default=1,
              help='Number of probe modes.')
@click.option('--perror', type=click.INT, default=0)
@click.option('--recover-positions', is_flag=True,
              help='Whether or not to correct positions.')
@click.option('--log-file', type=click.STRING, default='tike.log',
              help='Target log file.')
@click.option( '--folder', default='.', 
    type=click.Path(exists=False, file_okay=False, dir_okay=True,
                    writable=True, readable=True, resolve_path=True),
    help='A folder where the output is saved.')
def main(ifile, algorithm, niter, output_freq, 
          ngpu, gpu_id, auto_pin, gpu_mem_threshold, model, use_mpi, 
          recover_probe, recover_psi, overwrite, nmodes, folder,
          recover_positions, perror, log_file):
    """Reconstruct the DATASET using ALGORITHM and a probe with NMODES."""
    thismodule = sys.modules[__name__]
    hostname = socket.gethostname()

    logging.basicConfig(filename=log_file, #"/home/bicer/projects/tike/scripts/logs/tike.log",
                        filemode='a',
                        format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    logging.info(f"Passed parameters: \n"
                 f"ifile:{ifile}; ofile:{folder};\n"
                 f"algorithm:{algorithm}; iterations:{niter};\n"
                 f"output_freq:{output_freq}; ngpu:{ngpu};\n"
                 f"model:{model}; use_mpi:{use_mpi};\n"
                 f"recover_probe:{recover_probe}; recover_psi:{recover_psi};\n"
                 f"overwrite:{overwrite}; nmodes:{nmodes};\n"
                 f"recover_positions:{recover_positions}; perror:{perror};\n"
                 f"auto_pin:{auto_pin}; gpu_mem_threshold:{gpu_mem_threshold};\n"
                 f"gpu_id:{gpu_id}; log_file:{log_file}\n"
                 f"hostname:{hostname}\n")

    if auto_pin:
      avail_gpus = list_of_avail_gpus(gpu_mem_threshold)
      if len(avail_gpus) > 0:
        gpu_id = avail_gpus[0]
        cupy.cuda.Device(gpu_id).use()
        logging.info(f"Auto pinned task to GPU:{gpu_id}")
      else: 
        logging.error(f"No GPU is available, unable to pin task.")
        sys.exit(-1)

    if (not auto_pin) and (gpu_id is not None):
      avail_gpus = list_of_avail_gpus(gpu_mem_threshold)
      if ngpu > 1:
        logging.error(f"GPU id is set to {gpu_id}, currently the number of GPUs cannot be larger than 1.")
        sys.exit(-2)
      if gpu_id not in avail_gpus:
        logging.error(f"gpu_id:{gpu_id} is not available. Available gpus:{avail_gpus}.")
        sys.exit(-3)
      cupy.cuda.Device(gpu_id).use()
      logging.info(f"Pinned task to GPU:{gpu_id}")


    scan, data, probe = read_data_catalyst(ifile) 
    logging.info(f"File was read.")

    assert scan.ndim == 3 and scan.shape[-1] == 2
    assert data.ndim == 4
    assert scan.shape[0:2] == data.shape[0:2]
    assert probe.ndim == 6, probe.shape
    assert scan.shape[0] == probe.shape[0]

    # probe = np.load(f'{folder}/P_complex-{nmodes}-{algorithm}.npy')
    # psi = np.load(f'{folder}/O_complex-{nmodes}-{algorithm}.npy')

    scan, data, probe = clip_data(scan, data, probe)
    probe = add_modes_random_phase(probe, nmodes)

    assert scan.ndim == 3 and scan.shape[-1] == 2
    assert data.ndim == 4
    assert scan.shape[0:2] == data.shape[0:2]
    assert probe.ndim == 6
    assert scan.shape[0] == probe.shape[0]

    result = {'probe': probe, 'scan': scan}

    if os.path.exists(folder) and overwrite: 
      logging.info(f"Output folder exists; removing: {folder}")
      shutil.rmtree(folder)
    if not os.path.exists(folder): 
        logging.info(f"Creating output folder: {folder}")
        os.makedirs(folder)
    np.save(f'{folder}/scan-{nmodes}-{algorithm}-000.npy', result['scan'])

    for m in range(nmodes):
        skimage.io.imsave(
            f'{folder}/Px-{nmodes}-{m}-{algorithm}.tiff',
            np.square(np.abs(result['probe'][:, :, :, m])).astype('float32'))

    logging.info(f"Starting iterations.")
    check = output_freq
    for i in range(check, niter + check, check):

        if os.path.isfile(f'{folder}/O-{nmodes}-{algorithm}-{i:03d}.tiff'):
            result = {
                'psi': np.load(f'{folder}/O_complex-{nmodes}-{algorithm}.npy'),
                'probe':
                np.load(f'{folder}/P_complex-{nmodes}-{algorithm}.npy'),
                'scan':
                np.load(f'{folder}/scan-{nmodes}-{algorithm}-{i:03d}.npy'),
            }
            logging.info(f"skipped iteration {i}")
        else:
            result = tike.ptycho.reconstruct(
                **result,
                data=data,
                algorithm=algorithm,
                num_gpu=ngpu,
                num_iter=check,
                recover_psi=recover_psi,
                recover_probe=recover_probe,  #True if i > 2 * check else False,
                recover_positions=recover_positions,  #update_positions if i > check else False,
                rtol=-1,
                model=model,
                use_mpi=use_mpi
            )

            phase = np.angle(result['psi'])[0]
            ampli = np.abs(result['psi'])[0]
            phase = skimage.restoration.unwrap_phase(phase).astype('float32')
            skimage.io.imsave(f'{folder}/O-{nmodes}-{algorithm}-{i:03d}.tiff',
                              phase.astype('float32'))
            skimage.io.imsave(f'{folder}/A-{nmodes}-{algorithm}-{i:03d}.tiff',
                              ampli.astype('float32'))

            for m in range(nmodes):
                skimage.io.imsave(
                    f'{folder}/P-{nmodes}-{m}-{algorithm}-{i:03d}.tiff',
                    np.square(np.abs(probe[:, :, :, m])).astype('float32'))

            np.save(f'{folder}/O_complex-{nmodes}-{algorithm}.npy',
                    result['psi'])
            np.save(f'{folder}/P_complex-{nmodes}-{algorithm}.npy',
                    result['probe'])
            np.save(f'{folder}/scan-{nmodes}-{algorithm}-{i:03d}.npy',
                    result['scan'])

    logging.info(f"End of iterations.")


if __name__ == '__main__':
    main()
