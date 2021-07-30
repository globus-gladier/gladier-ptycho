def ptycho(data):
    """Test the ptycho tool"""
    import os
    import subprocess
    import logging
    from subprocess import PIPE
    
    logging.basicConfig(filename="/home/bicer/projects/tike/scripts/logs/funcx-ptycho-func.log",
                        filemode='a',
                        format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')

    
    
    logging.info("Starting ptycho funcx function.")
    
    python_path = data['python_path']
    script_path = data['script_path']
    
    #recon. script parameters
    ifpath = data['ifpath']
    ofpath = data['ofpath']
    rec_alg = data['rec_alg']
    rec_nmodes = data['rec_nmodes']
    rec_niter = data['rec_niter']
    rec_output_freq = data['rec_output_freq']
    rec_recover_psi = '--recover-psi' if (('rec_recover_psi' in data) and data['rec_recover_psi']) else ''
    rec_recover_probe = '--recover-probe' if (('rec_recover_probe' in data) and data['rec_recover_probe']) else ''
    rec_recover_positions = '--recover-positions' if (('rec_recover_positions' in data) and data['rec_recover_positions']) else ''
    rec_model = data['rec_model']
    rec_ngpu = data['rec_ngpu']
    rec_use_mpi = '--use-mpi' if (('rec_use_mpi' in data) and data['rec_use_mpi']) else ''
    rec_overwrite = '--overwrite' if (('rec_overwrite' in data) and data['rec_overwrite']) else ''
    rec_auto_pin = '--auto_pin' if (('rec_auto_pin' in data) and data['rec_auto_pin']) else ''
    
    try:
        os.mkdir(ofpath)
    except:
        pass
    
    cmd = f"{python_path} {script_path} --algorithm={rec_alg} --nmodes={rec_nmodes} --niter={rec_niter} --output-freq={rec_output_freq} {rec_recover_psi} {rec_recover_probe} {rec_recover_positions} --model={rec_model} --ngpu={rec_ngpu} {rec_use_mpi} --ifile='{ifpath}' {rec_overwrite} --folder='{ofpath}'"
    logging.info(f"Running command: {cmd}")
    # python /home/bicer/projects/tike/scripts/mpi_tike-recon.py --algorithm='cgrad' --nmodes=8 --niter=300 --output-freq=50 --recover-psi --recover-probe --overwrite --model='gaussian' --ngpu=8 --ifile='/grand/hp-ptycho/bicer/ptycho/comm_33IDD/globus_automate/input/300/extracted_scan300.h5' --folder='/grand/hp-ptycho/bicer/ptycho/comm_33IDD/globus_automate/output/300/'
    try:
        res = subprocess.run(cmd, stdout=PIPE, stderr=PIPE,
                             shell=True, executable='/bin/bash')
    except:
        pass
    outstr = f"{res.stdout}"
    return outstr
    
func_ptycho_uuid = fxc.register_function(ptycho)
print(func_ptycho_uuid)