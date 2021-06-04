payload = {'pathname': '/lus/theta-fs0/projects/ExaLearn/rchard/test_ptycho/extracted_scan350.h5',
           'output_dir': '/lus/theta-fs0/projects/ExaLearn/rchard/test_ptycho/output/recons/06/'}
tutorial_endpoint = '2eb0d751-46f4-4547-be1a-5d7150084623' # Public tutorial endpoint
res = fxc.run(payload, endpoint_id=tutorial_endpoint, function_id=func_uuid)
print(res)



prisma_fx_endpoint = '4bf59543-3398-42d2-9416-c628e9f5635f'

src_wf_root_path = '/prisma-data1/bicer/workflow' #'/gdata/RAVEN/bicer/2020-1/comm_33IDD/globus_automate'
src_input_folder_prefix = "input"
src_output_folder_prefix = "output"

dest_wf_root_path = '/grand/hp-ptycho/bicer/ptycho/comm_33IDD/globus_automate'
dest_input_folder_prefix = "input"
dest_output_folder_prefix = "output"


#src_input_folder_paths_regex = f"{src_wf_root_path}/{src_input_folder_prefix}/*.h5"
src_input_folder_paths_regex = f"{src_wf_root_path}/{src_input_folder_prefix}/*"
rid = fxc.run(src_input_folder_paths_regex, 
              endpoint_id=prisma_fx_endpoint, 
              function_id=fx_func_get_folder_paths_uuid)
src_input_folder_paths = fxc.get_result(rid)

src_output_folder_paths = []
dest_output_folder_paths = []
dest_input_folder_paths = []
for src_input_folder_path in src_input_folder_paths:
    #print(src_input_folder_path)
    iid = re.findall(r'\d+', src_input_folder_path)
    src_output_folder_path = f"{src_wf_root_path}/{src_output_folder_prefix}/{iid[-1]}"
    src_output_folder_paths.append(src_output_folder_path)
    dest_input_folder_path = f"{dest_wf_root_path}/{dest_input_folder_prefix}/{iid[-1]}"
    dest_input_folder_paths.append(dest_input_folder_path)
    dest_output_folder_path = f"{dest_wf_root_path}/{dest_output_folder_prefix}/{iid[-1]}"
    dest_output_folder_paths.append(dest_output_folder_path)
    
    

# src_input_folder_paths: diffraction patterh files to be processed @ APS
# src_output_folder_paths: folders for reconstrcuted images after processing @ APS
# dest_input_folder_paths: diffraction patterh files to be processed @ ALCF
# dest_output_folder_paths: folders for reconstrcuted images after processing @ ALCF

for (src_input_folder_path, src_output_folder_path, dest_input_folder_path, dest_output_folder_path ) in zip(src_input_folder_paths, src_output_folder_paths, dest_input_folder_paths, dest_output_folder_paths):
    print(f"Source input folder: {src_input_folder_path}; Source output folder: {src_output_folder_path}")
    print(f"Dest. input folder: {dest_input_folder_path}; Dest. output folder: {dest_output_folder_path}")
    print()


    theta_fx_endpoint = 'f765db7a-038c-47ea-9176-d81de31c054f' #'7f42390d-849a-42a7-905c-db6b22af28f7' # #FuncX endpoint

#src_endpoint = '9c9cb97e-de86-11e6-9d15-22000a1e3b52' #'aps#data' # Voyager
src_endpoint = 'dd916908-0072-11e7-badc-22000b9a448b' #'hostel' #aps/workstation
#dest_endpoint = '08925f04-569f-11e7-bef8-22000b9a448b' #'alcf#dtn_theta' # Theta DTN # 'e09e65f5-6d04-11e5-ba46-22000b92c6ec' #
dest_endpoint = 'e09e65f5-6d04-11e5-ba46-22000b92c6ec' #'alcf#dtn_mira'

# Ptycho recon params
script_path = '/home/bicer/projects/tike/scripts/tike-pinned-ptycho-recon.py'
python_path = "/home/bicer/projects/tyler/bin/python"

rec_alg = 'cgrad'
rec_nmodes = 8
rec_upd_pos = False
rec_niter = 100
rec_output_freq = 10
rec_recover_psi = True
rec_recover_probe= True
rec_recover_positions = False
rec_model = 'gaussian'
rec_ngpu = 1
rec_use_mpi = False
rec_overwrite = True
rec_auto_pin = True


flow_inputs = []

for (src_input_folder_path, src_output_folder_path, 
     dest_input_folder_path, dest_output_folder_path ) in zip(
    src_input_folder_paths, src_output_folder_paths, 
    dest_input_folder_paths, dest_output_folder_paths):
    
    flow_input = {
        "input": {
            "source_endpoint": f"{src_endpoint}",
            "source_path": f"{src_input_folder_path}",
            "dest_endpoint": dest_endpoint,
            "dest_path": f"{dest_input_folder_path}",

            "result_path": f"{dest_output_folder_path}",#f"{dest_resultpath}/out_{src_filename}",
            "source_result_path": f"{src_output_folder_path}", #/out_{src_filename}",
            "fx_ep": f"{theta_fx_endpoint}",
            "fx_id": f"{func_ptycho_uuid}",
            "params": {'ifpath': f"{dest_input_folder_path}",
                       'ofpath': f"{dest_output_folder_path}/",
                       'script_path': script_path,
                       'python_path': python_path,
                       'rec_alg': rec_alg,
                       'rec_nmodes': rec_nmodes,
                       'rec_upd_pos': rec_upd_pos,
                       'rec_niter': rec_niter,
                       'rec_output_freq': rec_output_freq,
                       'rec_recover_psi': rec_recover_psi,
                       'rec_recover_probe': rec_recover_probe,
                       'rec_recover_positions': rec_recover_positions,
                       'rec_model': rec_model,
                       'rec_ngpu': rec_ngpu,
                       'rec_use_mpi': rec_use_mpi,
                       'rec_overwrite': rec_overwrite,
                       'rec_auto_pin': rec_auto_pin}
        }
    }
    flow_inputs.append(flow_input)

#print(f"transfer file from {src_endpoint}#{src_filepath}/{src_filename} to {dest_endpoint}#{dest_filepath}/")
#print(f"recon file:{dest_filepath}/{src_filename} output:{dest_resultpath}/")
#print(f"transfer file from {dest_endpoint}:{dest_resultpath} to {src_endpoint}#{src_result_path}/")

MAX_QSIZE = 8
q0 = deque()
q1 = deque()

for i in range(len(flow_inputs)):
    if i<MAX_QSIZE:
        flow_action = flows_client.run_flow(flow_id, flow_scope, flow_inputs[i])
        q1.append(flow_action)
        print(f"Flow {i} initiated and added to q1: {flow_action['action_id']}")
    else: 
        q0.append(flow_inputs[i])
        print(f"Flow input {i} added to q0")

i=-1
while len(q1)>0:
    time.sleep(5)
    i = (i+1)%len(q1)
    flow = flows_client.flow_action_status(flow_id, flow_scope, q1[i]['action_id'])
    print(f"len(q0)={len(q0)}; len(q1)={len(q1)}; i={i}")
    print(f"Flow {i} status: {q1[i]['action_id']}: {flow['status']}")
    if flow['status'] == 'SUCCEEDED':
        del q1[i]
        if len(q0)>0:
            flow_input = q0.popleft()
            flow_action = flows_client.run_flow(flow_id, flow_scope, flow_input)
            q1.append(flow_action)
            print(f"New flow initiated and added to the q1: {flow_action['action_id']}")
            print(f"Copy from {flow_input['input']['params']['ifpath']} to {flow_input['input']['params']['ofpath']}")
