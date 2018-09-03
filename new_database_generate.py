import re
from os import listdir
import dexnet
from dexnet.database import mesh_processor
from perception.object_render import RenderMode
from meshpy import ObjFile
from autolab_core.yaml_config import YamlConfig
import numpy as np
from meshpy.mesh_renderer import *
from dexnet.database.database import Hdf5Database
from dexnet.constants import *



# DEFAULT_CONFIG = 'cfg/apps/cli_parameters.yaml'
SUPPORTED_MESH_FORMATS = ['.obj', '.off', '.wrl', '.stl']
RE_SPACE = re.compile('.*\s+$', re.M)
MAX_QUEUE_SIZE = 1000


dexnet_api = dexnet.DexNet()


config = dexnet_api._get_config()


objfilepath = '/home/kx/Desktop/example001/'
database_path = '/home/kx/dex-net2.0-custom/src/dex-net/data/test/database/example_prob.hdf5'
cache_dir = '/home/kx/Desktop/cache_dir/'       # path um temper _proc.obj zu store
dataset_name = 'mini_dexnet'
CONFIG_NAME = '/home/kx/dex-net2.0-custom/src/dex-net/test/config.yaml'


filename_list = listdir(objfilepath)
for i in range(len(filename_list)):
    obj_file_name = filename_list[i]
    obj_name = obj_file_name[:-4]
    filename = objfilepath + obj_file_name
    mp = mesh_processor.MeshProcessor(filename, cache_dir)
    # filename : :obj:`str`
    #         name of the mesh file to process
    # cache_dir : :obj:`str`
    #         directory to store intermediate files to

    mp._load_mesh()
    mp.mesh_.density = config['obj_density']
    mp._clean_mesh(config['obj_target_scale'], config['obj_scaling_mode'], config['use_uniform_com'],
                               rescale_mesh=config['rescale_objects'])
    mp._generate_sdf(config['path_to_sdfgen'], config['sdf_dim'], config['sdf_padding'])
    i += 1



objfilepath = cache_dir
filename_list = listdir(objfilepath)
obj_file_list = []
for filename in filename_list:
    if filename.endswith('obj'):
        obj_file_list.append(filename)

for i in range(len(obj_file_list)):

    # DEFAULT_CONFIG = 'cfg/apps/cli_parameters.yaml'
    SUPPORTED_MESH_FORMATS = ['.obj', '.off', '.wrl', '.stl']
    RE_SPACE = re.compile('.*\s+$', re.M)
    MAX_QUEUE_SIZE = 1000
    obj_file_name = obj_file_list[i]
    obj_name = obj_file_name[:-4]
    add_obj_path = objfilepath + obj_file_name

    dexnet_api = dexnet.DexNet()

    dexnet_api.open_database(database_path, create_db=True)
    dexnet_api.open_dataset(dataset_name, config=None, create_ds=True)

    dexnet_api.add_object(add_obj_path)
    dexnet_api.dataset.object_keys

    dexnet_api.compute_simulation_data(obj_name)
    dexnet_api.sample_grasps(object_name=obj_name, gripper_name='yumi_metal_spline')
    config = dexnet_api._get_config()

    dexnet_api.dataset.stable_pose_data(obj_name)
    dexnet_api.dataset.stable_poses(obj_name)

    dexnet_api.compute_metrics(metric_name='robust_ferrari_canny',
                               object_name= obj_name,
                               gripper_name='yumi_metal_spline')

    dexnet_api.compute_metrics(metric_name='force_closure',
                               object_name= obj_name,
                               gripper_name='yumi_metal_spline')

    # dexnet_api.compute_metadata(obj_name, config=None, overwrite=False)

    # try mesh_image
    database = Hdf5Database(database_path, access_level=READ_WRITE_ACCESS)
    dataset = dexnet_api.database.dataset(dataset_name)

    # read / write meshing
    obj = dexnet_api.dataset[dataset.object_keys[i]]
    mesh_filename = dataset.obj_mesh_filename(obj.key, overwrite=True)
    f = ObjFile(mesh_filename)
    load_mesh = f.read()

    write_vertices = np.array(obj.mesh.vertices)
    write_triangles = np.array(obj.mesh.triangles)
    load_vertices = np.array(load_mesh.vertices)
    load_triangles = np.array(load_mesh.triangles)

    # test rendering images
    print('test rendering images')
    stable_poses = dataset.stable_poses(obj.key)

    for j in range(len(stable_poses)):
        stable_pose = stable_poses[j]

        # setup virtual camera
        print('setup virtual camera')
        CONFIG = YamlConfig(CONFIG_NAME)
        width = CONFIG['width']
        height = CONFIG['height']
        f = CONFIG['focal']
        cx = float(width) / 2
        cy = float(height) / 2
        ci = CameraIntrinsics('camera', fx=f, fy=f, cx=cx, cy=cy,
                              height=height, width=width)
        vp = ViewsphereDiscretizer(min_radius=CONFIG['min_radius'],
                                   max_radius=CONFIG['max_radius'],
                                   num_radii=CONFIG['num_radii'],
                                   min_elev=CONFIG['min_elev'] * np.pi,
                                   max_elev=CONFIG['max_elev'] * np.pi,
                                   num_elev=CONFIG['num_elev'],
                                   num_az=CONFIG['num_az'],
                                   num_roll=CONFIG['num_roll'])
        vc = VirtualCamera(ci)

        # render segmasks and depth
        print('render segmasks and depth')
        render_modes = [RenderMode.SEGMASK, RenderMode.DEPTH, RenderMode.SCALED_DEPTH]
        for render_mode in render_modes:
            rendered_images = vc.wrapped_images_viewsphere(obj.mesh, vp,
                                                           render_mode,
                                                           stable_pose)
            pre_store_num_images = len(rendered_images)
            dataset.store_rendered_images(obj.key, rendered_images,
                                          stable_pose_id=stable_pose.id,
                                          render_mode=render_mode)
            rendered_images = dataset.rendered_images(obj.key,
                                                      stable_pose_id=stable_pose.id,
                                                      render_mode=render_mode)
        j+=1

    i+=1