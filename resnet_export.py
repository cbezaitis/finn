import torch
import onnx
from finn.util.basic import make_build_dir
from finn.util.visualization import showInNetron
import os
from brevitas.export import export_qonnx
from qonnx.util.cleanup import cleanup as qonnx_cleanup
from qonnx.util.cleanup import cleanup_model
from qonnx.core.modelwrapper import ModelWrapper
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, RemoveStaticGraphInputs
build_dir = os.environ["FINN_BUILD_DIR"]
# from finn.util.test import get_test_model_trained
from qonnx.transformation.infer_datatypes import InferDataTypes
from finn.transformation.streamline import Streamline
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
import finn.transformation.streamline.absorb as absorb
from finn.transformation.streamline.reorder import MakeMaxPoolNHWC, MoveScalarLinearPastInvariants
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.general import RemoveUnusedTensors
from qonnx.transformation.extract_conv_bias import ExtractBiasFromConv
from qonnx.transformation.base import Transformation
from qonnx.transformation.extract_conv_bias import ExtractBiasFromConv
from qonnx.transformation.gemm_to_matmul import GemmToMatMul
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.quant_constant_folding import FoldTransposeIntoQuantInit
from qonnx.transformation.remove import RemoveIdentityOps
from finn.util.pytorch import ToTensor,NormalizePreProc,Normalize
from qonnx.transformation.insert_topk import InsertTopK
from qonnx.transformation.batchnorm_to_affine import BatchNormToAffine
from finn.transformation.qonnx.fold_quant_weights import FoldQuantWeights
from finn.transformation.qonnx.infer_quant_avg_pool_2d import (
    AvgPoolAndTruncToQuantAvgPool,
)
from finn.transformation.streamline.collapse_repeated import CollapseRepeatedMul, CollapseRepeatedAdd
from finn.transformation.qonnx.quant_act_to_multithreshold import (
    ConvertQuantActToMultiThreshold,
    default_filter_function_generator,
)
from qonnx.core.datatype import DataType
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from finn.transformation.streamline.reorder import (
    MoveOpPastFork,
    MoveLinearPastEltwiseAdd,
    MoveScalarMulPastConv,
    MoveScalarAddPastMatMul,
    MoveScalarAddPastConv,
    MoveAddPastConv,
    MoveAddPastMul,
    MoveScalarLinearPastInvariants,
    MoveScalarMulPastMatMul,
)
from finn.transformation.streamline.absorb import (
    AbsorbAddIntoMultiThreshold,
    AbsorbMulIntoMultiThreshold,
    FactorOutMulSignMagnitude,
    Absorb1BitMulIntoConv,
    AbsorbScalarMulAddIntoTopK,
    AbsorbTransposeIntoMultiThreshold,  
)
from qonnx.transformation.general import (
    ConvertDivToMul,
    ConvertSubToAdd,
)
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from qonnx.core.datatype import DataType

filter_function=default_filter_function_generator(max_multithreshold_bit_width=8)
import pkg_resources as pk
import numpy as np
from qonnx.transformation.gemm_to_matmul import GemmToMatMul

# import finn.core.onnx_exec as oxe
from qonnx.core.onnx_exec import execute_onnx





# Wrap model with a correct path 
model = ModelWrapper("/home/babis/phd/code/finn/quant_resnet18_w4a4_a2q_plus_32b-60b3176e.onnx")
model = cleanup_model(model)
model = model.transform(InferShapes())
model = model.transform(FoldConstants())
model = model.transform(GiveUniqueNodeNames())
model = model.transform(GiveReadableTensorNames())
model = model.transform(RemoveStaticGraphInputs())
# model = model.transform(InferDataTypes())
# model = model.transform(ExtractBiasFromConv())

model = model.transform(ConvertQONNXtoFINN())
model = model.transform(InferShapes())
model = model.transform(FoldConstants())
model = model.transform(GiveUniqueNodeNames())
model = model.transform(GiveReadableTensorNames())
model = model.transform(RemoveStaticGraphInputs())
model = model.transform(InferDataTypes())

model.save(build_dir + "/step00.onnx")
# model = model.transform(InferShapes())
# model = model.transform(FoldConstants())
# model = model.transform(GiveUniqueNodeNames())
# model = model.transform(GiveReadableTensorNames())
# model = model.transform(InferDataTypes())
# model = model.transform(RemoveStaticGraphInputs())
# model.save(build_dir + "/end2end_cnv_w1a1_tidy.onnx")


# from finn.util.pytorch import ToTensor
# from qonnx.transformation.merge_onnx_models import MergeONNXModels
# from qonnx.core.datatype import DataType

# # model = ModelWrapper(build_dir+"/end2end_cnv_w1a1_tidy.onnx")
global_inp_name = model.graph.input[0].name
ishape = model.get_tensor_shape(global_inp_name)
# preprocessing: torchvision's ToTensor divides uint8 inputs by 255
# mean = [0.491, 0.482, 0.447]
# std = [0.247, 0.243, 0.262]
# std = 0.250
totensor_pyt = ToTensor()
chkpt_preproc_name = build_dir+"/end2end_cnv_w1a1_preproc.onnx"
export_qonnx(totensor_pyt, torch.randn(ishape), chkpt_preproc_name)
qonnx_cleanup(chkpt_preproc_name, out_file=chkpt_preproc_name)
pre_model = ModelWrapper(chkpt_preproc_name)
pre_model.set_tensor_datatype(pre_model.graph.input[0].name, DataType["UINT8"])
pre_model = pre_model.transform(InferShapes())
pre_model = pre_model.transform(InferDataLayouts())
pre_model = pre_model.transform(FoldConstants())
pre_model = pre_model.transform(GiveUniqueNodeNames())
pre_model = pre_model.transform(RemoveUnusedTensors())
pre_model = pre_model.transform(RemoveStaticGraphInputs())
pre_model = pre_model.transform(GiveReadableTensorNames())
pre_model = pre_model.transform(ConvertQONNXtoFINN())
# pre_model = qonnx_cleanup(pre_model)
pre_model.save( build_dir + "/pre_model.onnx")

# # # join preprocessing and core model
model = model.transform(MergeONNXModels(pre_model))
# # add input quantization annotation: UINT8 for all BNN-PYNQ models
model.set_tensor_datatype(global_inp_name, DataType["UINT8"])
# model = model.transform(Streamline())
# # # postprocessing: insert Top-1 node at the end
model = model.transform(InsertTopK(k=1))
model = model.transform(InferShapes())
model = model.transform(InferDataLayouts())
model = model.transform(FoldConstants())
model = model.transform(GiveUniqueNodeNames())
model = model.transform(RemoveUnusedTensors())
model = model.transform(RemoveStaticGraphInputs())
model = model.transform(GiveReadableTensorNames())

model = model.transform(ConvertQONNXtoFINN())
# model = model.transform(CollapseRepeatedAdd())

streamline_transformations = [
    MoveOpPastFork(['Mul']),
    MoveLinearPastEltwiseAdd(),
    ConvertDivToMul(),
    ConvertSubToAdd(),
    BatchNormToAffine(),
    MoveScalarMulPastConv(),
    MoveScalarLinearPastInvariants(),
    MoveScalarMulPastMatMul(),
    MoveLinearPastEltwiseAdd(),
    CollapseRepeatedMul(),
    FactorOutMulSignMagnitude(),
    Absorb1BitMulIntoConv(),
    RemoveIdentityOps(),
    InferDataTypes(),
    RoundAndClipThresholds(),
]

# Run all streamlining steps 

model.save(build_dir +  "/step01.onnx")
for t in streamline_transformations:
    model = model.transform(t)
    model = cleanup_model(model)
model.save(build_dir +  "/step02.onnx")
model = model.transform(MoveScalarAddPastConv())
model = model.transform(MoveScalarMulPastConv())
model = model.transform(MoveAddPastMul())
model = model.transform(MoveLinearPastEltwiseAdd())
model.save(build_dir +  "/step03.onnx")
model = model.transform(CollapseRepeatedMul())
model = model.transform(CollapseRepeatedAdd())
model.save(build_dir +  "/step04.onnx")
model = model.transform(AbsorbAddIntoMultiThreshold())
model = model.transform(InferDataTypes())
model = model.transform(RoundAndClipThresholds())
model = cleanup_model(model)
model.save(build_dir +  "/step05.onnx")

model = model.transform(AbsorbMulIntoMultiThreshold())
model = model.transform(InferDataTypes())
model = model.transform(RoundAndClipThresholds())
model = cleanup_model(model)
model.save(build_dir +  "/step06.onnx")

model = model.transform(AbsorbMulIntoMultiThreshold())
model = model.transform(RoundAndClipThresholds())
model = cleanup_model(model)
model = model.transform(MoveAddPastMul())
model = cleanup_model(model)
model = model.transform(CollapseRepeatedMul())
model = cleanup_model(model)
model = model.transform(AbsorbAddIntoMultiThreshold())
model = model.transform(RoundAndClipThresholds())
model = cleanup_model(model)

model = cleanup_model(model)
model = model.transform(AbsorbAddIntoMultiThreshold())
model = cleanup_model(model)
model = model.transform(AbsorbMulIntoMultiThreshold())
model = cleanup_model(model)
model = model.transform(AbsorbMulIntoMultiThreshold())
model = cleanup_model(model)
model = model.transform(LowerConvsToMatMul())
model = cleanup_model(model)    
model = model.transform(AbsorbMulIntoMultiThreshold())
model = model.transform(InferShapes())
model = model.transform(InferDataLayouts())
model = model.transform(FoldConstants())
model = model.transform(GiveUniqueNodeNames())
model = model.transform(RemoveUnusedTensors())
model = model.transform(RemoveStaticGraphInputs())
model = model.transform(GiveReadableTensorNames())
model = model.transform(InferDataTypes())
model = model.transform(RoundAndClipThresholds())
model = cleanup_model(model)
model = model.transform(AbsorbScalarMulAddIntoTopK())
model = cleanup_model(model)
model = model.transform(AbsorbMulIntoMultiThreshold())
model = cleanup_model(model)
model = model.transform(RoundAndClipThresholds())
model = cleanup_model(model)
model.set_tensor_datatype("Add_8_param0", DataType["FLOAT32"])
model.save(build_dir + "/quant_resnet18_w4a4_a2q_plus_32b-60b3176e_streamlined.onnx")


ishape = (1, 3, 32, 32)
model = model.transform(InferShapes())
model = model.transform(FoldConstants())
assert len(model.graph.input) == 1
assert len(model.graph.output) == 1

## Later maybe
# fn = pk.resource_filename("finn.qnn-data", "cifar10/cifar10-test-data-class3.npz")
# input_tensor = np.load(fn)["arr_0"].astype(np.float32)
# # print(fn.get_resource_filename())
# # input_tensor = input_tensor / 255
# assert input_tensor.shape == (1, 3, 32, 32)
# # run using FINN-based execution
# input_dict = {model.graph.input[0].name: input_tensor}
# output_dict = execute_onnx(model, input_dict, True)
# produced = output_dict[model.graph.output[0].name]
# print("Produced")
# print(produced)
# # do forward pass in PyTorch/Brevitas
# input_tensor = torch.from_numpy(input_tensor).float()
# # expected = cnv.forward(input_tensor).detach().numpy()
# # print("Expected")
# # print(expected)
# # assert np.isclose(produced, expected, atol=1e-3).all()
# assert produced == 3
