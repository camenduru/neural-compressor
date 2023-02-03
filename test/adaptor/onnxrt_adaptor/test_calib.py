import os
import shutil
import unittest
import onnxruntime as ort
import torch
import torchvision
import onnx
import numpy as np
import sys
sys.path.append('/home/yuwenzho/calibration/calib_v1')
from collections import OrderedDict
from onnx import onnx_pb as onnx_proto
from onnx import helper, TensorProto, numpy_helper
from onnxruntime.quantization import quantize_static, QuantFormat, CalibrationMethod, CalibrationDataReader, QuantType
from neural_compressor.adaptor import FRAMEWORKS
from neural_compressor.data import Datasets, DATALOADERS
from neural_compressor.experimental import Quantization, common
from neural_compressor.experimental import Benchmark, common
from neural_compressor import options
from neural_compressor.adaptor.pytorch import get_torch_version
from neural_compressor import conf
from packaging.version import Version
import onnxruntime
np.random.seed(1)

class MatmulDataset:
    def __init__(self):
        self.data = []
        self.label = []
        for i in range(2):
            # self.data.append(np.ones((5,5)).astype('float32')) 
            # self.label.append(np.ones((5,1)).astype('float32'))
            self.data.append(np.random.randn(5,5).astype('float32')) 
            self.label.append(np.random.randn(5,1).astype('float32'))

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)

class ONNXRTMatmulDataset(CalibrationDataReader):
    def __init__(self, model, dataloader) -> None:
        self.dataset = []
        for data in dataloader:
            self.dataset.append(data)
        self.datasize = len(self.dataset)
        session = onnxruntime.InferenceSession(model.SerializeToString(), None)
        self.input_name = [i.name for i in session.get_inputs()]
        self.enum_data_dicts = iter([dict(zip(self.input_name, data)) for data in self.dataset])

    def get_next(self):
        return next(self.enum_data_dicts, None)

def build_matmul_model():
    A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [1, 5, 5])
    C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [1, 5, 2])
    D = helper.make_tensor_value_info('D', TensorProto.FLOAT, [1, 5, 2])
    H = helper.make_tensor_value_info('H', TensorProto.FLOAT, [1, 5, 2])

    e_value = np.random.randint(1,2, size=(10)).astype(np.float32)
    B_init = helper.make_tensor('B', TensorProto.FLOAT, [5, 2], e_value.reshape(10).tolist())
    E_init = helper.make_tensor('E', TensorProto.FLOAT, [1, 5, 2], e_value.reshape(10).tolist())

    matmul_node = onnx.helper.make_node('MatMul', ['A', 'B'], ['C'], name='Matmul')
    add = onnx.helper.make_node('Add', ['C', 'E'], ['D'], name='add')
 
    f_value = np.random.randint(1,2, size=(10)).astype(np.float32)
    F_init = helper.make_tensor('F', TensorProto.FLOAT, [1, 5, 2], e_value.reshape(10).tolist())
    add2 = onnx.helper.make_node('Add', ['D', 'F'], ['H'], name='add2')
 
    graph = helper.make_graph([matmul_node, add, add2], 'test_graph_1', [A], [H], [B_init, E_init, F_init])
    model = helper.make_model(graph)
    model = helper.make_model(graph, **{'opset_imports': [helper.make_opsetid('', 13)]})
    onnx.save(model, 'model.onnx')
    return model

class TestAdaptorONNXRT(unittest.TestCase):
    matmul_dataset = MatmulDataset()
    matmul_dataloader = DATALOADERS['onnxrt_qlinearops'](matmul_dataset, batch_size=1)

    @classmethod
    def setUpClass(self):
        self.matmul_model = build_matmul_model()

    def test_onnx(self):
        dr = ONNXRTMatmulDataset(self.matmul_model, self.matmul_dataloader)
        quantize_static('model.onnx', 'ort_model.onnx',
                        dr,
                        quant_format=QuantFormat.QOperator,
                        calibrate_method=CalibrationMethod.Entropy,
                        # calibrate_method=CalibrationMethod.Percentile,
                        # calibrate_method=CalibrationMethod.MinMax,
                        activation_type=QuantType.QUInt8,
                        weight_type=QuantType.QInt8,
                        per_channel=True,
                        # extra_options={'CalibTensorRangeSymmetric': False}
                            )

    def test_calib(self):
        from neural_compressor.utils.constant import FP32, INT8_SYM_MINMAX_PERTENSOR, UINT8_ASYM_MINMAX_PERTENSOR, UINT8_ASYM_KL_PERTENSOR
        conf.model.framework = 'onnxrt_qlinearops'
        conf.quantization.approach = 'post_training_static_quant'
        # conf.quantization.calibration.sampling_size = 1
        {'dtype': ['int8'],'scheme': ['sym'], 'algorithm': ['percentile'],'granularity': ['per_tensor']}

        conf.quantization.op_wise = {
            'Matmul': {'weight': INT8_SYM_MINMAX_PERTENSOR, 'activation': {'dtype': ['int8'],'scheme': ['sym'], 'algorithm': ['percentile'],'granularity': ['per_tensor']}},
            'add': {'weight': INT8_SYM_MINMAX_PERTENSOR, 'activation': {'dtype': ['int8'],'scheme': ['sym'], 'algorithm': ['entropy'],'granularity': ['per_tensor']}},
            'add2': {'weight': INT8_SYM_MINMAX_PERTENSOR, 'activation': {'dtype': ['int8'],'scheme': ['sym'], 'algorithm': ['minmax'],'granularity': ['per_tensor']}}}
        conf.evaluation.accuracy.metric = {'MSE': {'compare_label': False}}
        quantizer = Quantization(conf)
        quantizer.calib_dataloader = self.matmul_dataloader
        quantizer.eval_dataloader = self.matmul_dataloader
        quantizer.model = self.matmul_model
        q_model = quantizer.fit()
        q_model.save('q_model.onnx')

if __name__ == "__main__":
    unittest.main()