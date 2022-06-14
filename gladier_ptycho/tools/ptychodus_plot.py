from gladier import GladierBaseTool, generate_flow_definition

def plot_ptychodus(**data):
    import os
    import subprocess
    import numpy as np
    import matplotlib.pyplot as plt

    proc_dir = data['proc_dir']

    npz_file = 'ptychodus.npz'

    os.chdir(proc_dir) 

    data = np.load(npz_file)
    data_probe = data['probe']
    data_object = data['object']
    pixel_size = data['pixelSizeInMeters']

    plt.imsave('object_real.jpg', data_object.real)
    plt.imsave('object_im.jpg', data_object.imag)


    data['pilot']['dataset'] = data['upload_dir']
    data['pilot']['index'] = data['search_index']
    data['pilot']['project'] = data['search_project']
    data['pilot']['source_globus_endpoint'] = data['source_globus_endpoint'] ##Review this later
    data['pilot']['groups'] = data.get('groups',[])

    # Update any metadata in the pilot 'metadata' key
    metadata = data['pilot'].get('metadata', {})
    metadata.update(beamline_metadata)
    metadata.update({
        'sample_name': data['sample_name'],
        'pixel_size': pixel_size
    })
    data['pilot']['metadata'] = metadata

    return {
        'pilot': data['pilot']
    }


@generate_flow_definition(modifiers={
    'plot_ptychodus': {'WaitTime':7200}
})
class PtychodusPlot(GladierBaseTool):
    flow_input = {}
    required_input = [
        'proc_dir',
        'funcx_endpoint_compute',
    ]
    funcx_functions = [plot_ptychodus]
