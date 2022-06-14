from gladier import GladierBaseTool, generate_flow_definition

def ptychodus_fix_ini(**data):
    import os
    import subprocess

    data_dir = data['data_dir']
    proc_dir = data['proc_dir']

    ini_name = f"{proc_dir}/ptychodus.ini"

    os.chdir(proc_dir) 

    template_data = {
        'file_path': proc_dir+'fly001_master.h5',
        'pos_file': proc_dir+'fly001_pos.csv',
        'n_thread': 16,
    }

    template_ini = Template("""
    [Data]
FileType = Velociprobe
FilePath = $file_path
ScratchDirectory = /dev/null
NumberOfDataThreads = $n_thread

[Detector]
NumberOfPixelsX = 1030
PixelSizeXInMeters = 0.000075
NumberOfPixelsY = 514
PixelSizeYInMeters = 0.000075
DetectorDistanceInMeters = 2.335

[Crop]
CropEnabled = True
CenterXInPixels = 540
CenterYInPixels = 259
ExtentXInPixels = 128
ExtentYInPixels = 128

[Scan]
Initializer = FromFile
InputFileType = CSV
InputFilePath = $pos_file
ExtentX = 10
ExtentY = 10
StepSizeXInMeters = 0.000001
StepSizeYInMeters = 0.000001
JitterRadiusInMeters = 0
Transform = +X+Y

[Probe]
Initializer = FresnelZonePlate
InputFileType = NPY
InputFilePath = /path/to/probe.npy
AutomaticProbeSizeEnabled = True
ProbeSize = 128
ProbeEnergyInElectronVolts = 10000.0
ProbeDiameterInMeters = 0.000400
ZonePlateRadiusInMeters = 0.000090
OutermostZoneWidthInMeters = 5.0E-8
BeamstopDiameterInMeters = 0.000060
DefocusDistanceInMeters = 0.000800

[Object]
Initializer = Random
InputFileType = NPY
InputFilePath = /path/to/object.npy

[Reconstructor]
Algorithm = rPIE
OutputFileType = NPZ
OutputFilePath = ptychodus.npz

[PtychoPy]
ProbeModes = 1
Threshold = 0
ReconstructionIterations = 100
ReconstructionTimeInSeconds = 0
RMS = False
UpdateProbe = 10
UpdateModes = 20
PhaseConstraint = 1

[Tike]
UseMpi = False
NumGpus = 1
NoiseModel = gaussian
NumProbeModes = 5
NumBatch = 10
NumIter = 50
CgIter = 2
Alpha = 0.05
StepLength = 1

[TikePositionCorrection]
UseAdaptiveMoment = False
MDecay = 0.9
VDecay = 0.999
UsePositionCorrection = False
UsePositionRegularization = False

[TikeProbeCorrection]
UseAdaptiveMoment = False
MDecay = 0.9
VDecay = 0.999
UseProbeCorrection = True
OrthogonalityConstraint = True
CenteredIntensityConstraint = False
SparsityConstraint = 1
ProbeSupportWeight = 10
ProbeSupportRadius = 0.3
ProbeSupportDegree = 5

[TikeObjectCorrection]
UseAdaptiveMoment = False
MDecay = 0.9
VDecay = 0.999
UseObjectCorrection = True
PositivityConstraint = 0
SmoothnessConstraint = 0
    """)
    ini_data = template_ini.substitute(template_data)

    with open(ini_name, 'w') as fp:
        fp.write(ini_data)
    return ini_name 



    
    return 


@generate_flow_definition()
class PtychodusFixIni(GladierBaseTool):
    flow_input = {}
    required_input = [
        'proc_dir',
        'funcx_endpoint_compute',
    ]
    funcx_functions = [ptychodus_fix_ini]
