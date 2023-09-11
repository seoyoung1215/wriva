from .google_scanned_objects import *
from .realestate import *
from .deepvoxels import *
from .realestate import *
from .ibrnet_collected import *
from .realestate import *
from .spaces_dataset import *
from .nerf_synthetic import *
from .shiny import *
from .shiny_render import *
from .nerf_synthetic_render import *
from .nmr_dataset import *
from .mvimgnet import *
from .omniobject3d import *
from .wriva import *
from .wrivareal import *
from .wrivareal_render import *
from .nerds import *
from .nerds_sc import *

dataset_dict = {
    "spaces": SpacesFreeDataset,
    "google_scanned": GoogleScannedDataset,
    "realestate": RealEstateDataset,
    "deepvoxels": DeepVoxelsDataset,
    "nerf_synthetic": NerfSyntheticDataset,
    "ibrnet_collected": IBRNetCollectedDataset,
    "shiny": ShinyDataset,
    "shiny_render": ShinyRenderDataset,
    "nerf_synthetic_render": NerfSyntheticRenderDataset,
    "nmr": NMRDataset,
    "mvimgnet": MVImgnetDataset,
    "omni": OmniObject3DDataset,
    "wriva": WRIVADataset,
    "wriva_real": WRIVARealDataset,
    "wriva_real_render": WRIVARealRenderDataset,
    "nerds": NeRDS360Dataset,
    "nerds_sc": NeRDS360_SC_Dataset,
}
