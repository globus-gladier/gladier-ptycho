import datetime
import copy


class BaseDeployment:
    globus_endpoints = dict()
    funcx_endpoints = dict()
    flow_input = dict()

    def get_input(self):
        fi = self.flow_input.copy()
        fi['input'].update(self.funcx_endpoints)
        fi['input'].update(self.globus_endpoints)
        return fi

class TekinDeployment(BaseDeployment):

    globus_endpoints = {
        'globus_endpoint_source': 'fdc7e74a-fa78-11e8-9342-0e3d676669f4',
        'globus_endpoint_proc': '08925f04-569f-11e7-bef8-22000b9a448b',
    }

    funcx_endpoints = {
        'funcx_endpoint_non_compute': 'e449e8b8-e114-4659-99af-a7de06feb847',
        'funcx_endpoint_compute': '4c676cea-8382-4d5d-bc63-d6342bdb00ca',
    }

    flow_input = {
        'input': {
            'staging_dir': '/eagle/APSDataAnalysis/XPCS/data_online',
        }
    }
    
deployment_map = {
    'tekin-prod': TekinDeployment(),
}
