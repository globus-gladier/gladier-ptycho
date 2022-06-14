#!/usr/bin/env python

# Enable Gladier Logging
#import gladier.tests

import argparse
import os


from gladier import  GladierBaseClient, generate_flow_definition

from tools.ptychodus_exec import PtychodusExec
from tools.ptychodus_plot import PtychodusPlot

@generate_flow_definition(modifiers={
   'publish_gather_metadata': {'WaitTime': 240, 'payload': '$.PtychodusPlot.details.result[0].pilot'},
})
class PtychodusFlow(GladierBaseClient):
    gladier_tools = [
        'gladier_tools.globus.transfer.Transfer:FromStorage',
      PtychodusExec,
#        'gladier_tools.globus.transfer.Transfer:DataBack',
       PtychodusPlot,
       'gladier_tools.publish.Publish'
    ]
    


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--localdir', help='input file pathname',
                        default='/test_data/fly001')
    parser.add_argument('--datadir', help='Output Directory on theta',
                        default='/test_proc/')
    return parser.parse_args()


if __name__ == '__main__':

    args = arg_parse()

    # Experiment paths
    local_dir = args.localdir

    sample_name = 'test_raf1'

    data_dir = os.path.join(args.datadir,sample_name)
    # Base input for the flow
    flow_input = {
        'input': {
            # #Processing variables
            'sample_name': sample_name,
            'local_dir': local_dir,
            'data_dir': data_dir, #relative to endpoint
            'proc_dir': '/eagle/APSDataAnalysis/PTYCHO/'+data_dir, #relative to funcx
            
            # # globus local endpoint
            'from_storage_transfer_source_endpoint_id': '80150e2e-5e88-4d35-b3cd-170b25b60538',
            'from_storage_transfer_source_path': str(local_dir),

            # eagle endpoint for first transfer (may be repeated since it is eagle/eagle)
            'from_storage_transfer_destination_endpoint_id' : '80150e2e-5e88-4d35-b3cd-170b25b60538',
            'from_storage_transfer_destination_path': str(data_dir),
            'from_storage_transfer_recursive': True,


            # # funcX endpoints
            'funcx_endpoint_non_compute': 'e449e8b8-e114-4659-99af-a7de06feb847',
            'funcx_endpoint_compute': "a93b6438-6ff7-422e-a1a2-9a4c6d9c1ea5",


            'upload_dir':'/eagle/APSDataAnalysis/PTYCHO/portal/'+sample_name,
            'search_index': '93e343cc-b555-4d60-9aab-80ff191a8abb',
            'search_project': 'ptycho',
            'source_globus_endpoint': "80150e2e-5e88-4d35-b3cd-170b25b60538",
            'groups': [],
            'pilot':{} #this seem to be a bug on the autogeneration
        }
    }

    ptycho_flow = PtychodusFlow()
    
    run_label = 'Raf is testing'

    flow_run = ptycho_flow.run_flow(flow_input=flow_input, label=run_label)

    print('run_id : ' + flow_run['action_id'])



