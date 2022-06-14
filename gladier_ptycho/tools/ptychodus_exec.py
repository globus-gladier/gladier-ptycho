from gladier import GladierBaseTool, generate_flow_definition

def ptychodus_exec(**data):
    import os
    import subprocess

    data_dir = data['data_dir']
    proc_dir = data['proc_dir']

    ini_name = f"{proc_dir}/ptychodus.ini"

    timeout = data.get('timeout', 1200)

    logname = 'log-ptychodus.txt'

    cmd = f'timeout {timeout} ptychodus -b -s  {ini_name}  > {logname}'

    os.chdir(proc_dir) 

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             shell=True, executable='/bin/bash')
    
    return cmd, str(res.stdout), str(res.stderr)


@generate_flow_definition(modifiers={
    'ptychodus_exec': {'WaitTime':7200}
})
class PtychodusExec(GladierBaseTool):
    flow_input = {}
    required_input = [
        'proc_dir',
        'funcx_endpoint_compute',
    ]
    funcx_functions = [ptychodus_exec]
