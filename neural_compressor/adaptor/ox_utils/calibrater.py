import numpy as np
from neural_compressor.adaptor.ox_utils.util import smooth_distribution

CALIBRATION = {}

def calib_registry(calib_method):
    def decorator_calib(cls):
        assert cls.__name__.endswith(
            'Calibrater'), "The name of subclass of Calibrater should end with \'Calibrater\' substring."
        if cls.__name__[:-len('Calibrater')] in CALIBRATION: # pragma: no cover
            raise ValueError('Cannot have two operators with the same name.')
        CALIBRATION[calib_method.strip()] = cls
        return cls
    return decorator_calib


class CalibraterBase:
    def __init__(self):
        self.outputs_dict = {}

    def check_calib(self):
        for intermediate_output in self.intermediate_outputs:
            assert len(intermediate_output) == len(self.intermediate_outputs_names), \
                'intermediate_outputs length should equal to intermediate_outputs_names'

    def collect(self, intermediate_outputs, intermediate_outputs_names):
        self.intermediate_outputs = intermediate_outputs
        self.intermediate_outputs_names = intermediate_outputs_names
        self.check_calib()
        self.collect_calib_data()

@calib_registry(calib_method='minmax')
class MinMaxCalibrater(CalibraterBase):
    def __init__(self):
        super(MinMaxCalibrater, self).__init__()

    def collect_calib_data(self):
        for intermediate_output in self.intermediate_outputs:
            for (data, name) in zip(intermediate_output, self.intermediate_outputs_names):
                self.outputs_dict.setdefault(name, []).append([data.min(), data.max()])

@calib_registry(calib_method='percentile')
class PercentileCalibrater(CalibraterBase):
    def __init__(self, 
                 symmetric=True,
                 num_bins=2048,
                 percentile=99.999,):
        super(PercentileCalibrater, self).__init__()
        self.collector = None
        self.method = 'percentile'
        self.symmetric = symmetric
        self.num_bins = num_bins
        self.percentile = percentile
        print("Finding optimal threshold for each tensor using {} algorithm ...".format(self.method))
        print("Number of histogram bins : {}".format(self.num_bins))
        print("Percentile : ({},{})".format(100.0 - percentile, percentile))

    def collect_calib_data(self):
        merged_dict = {}
        for intermediate_output in self.intermediate_outputs:
            for (data, name) in zip(intermediate_output, self.intermediate_outputs_names):
                merged_dict.setdefault(name, []).append(data)
        if not self.collector:
            self.collector = HistogramCollector(
                method=self.method,
                symmetric=self.symmetric,
                num_bins=self.num_bins,
                percentile=self.percentile,
            )
        self.collector.collect(merged_dict)
        for name, collect_data in self.collector.compute_collection_result().items():
            self.outputs_dict[name] = [list(collect_data)]

@calib_registry(calib_method='entropy')
class EntropyCalibrater(CalibraterBase):
    def __init__(self, 
                 num_bins=128,
                 num_quantized_bins=128,):
        super(EntropyCalibrater, self).__init__()
        self.collector = None
        self.method = 'entropy'
        self.num_bins = num_bins
        self.num_quantized_bins = num_quantized_bins
        print("Finding optimal threshold for each tensor using {} algorithm ...".format(self.method))
        print("Number of histogram bins : {}" \
         "(The number may increase depends on the data it collects)".format(self.num_bins))
        print("Number of quantized bins : {}".format(self.num_quantized_bins))
    
    def collect_calib_data(self):
        merged_dict = {}
        for intermediate_output in self.intermediate_outputs:
            for (data, name) in zip(intermediate_output, self.intermediate_outputs_names):
                merged_dict.setdefault(name, []).append(data)
        if not self.collector:
            self.collector = HistogramCollector(
                method=self.method,
                num_bins=self.num_bins,
                num_quantized_bins=self.num_quantized_bins,
            )
        self.collector.collect(merged_dict)
        for name, collect_data in self.collector.compute_collection_result().items():
            self.outputs_dict[name] = [list(collect_data)]
    


class HistogramCollector:
    """
    Collecting histogram for each tensor. Percentile and Entropy method are supported.

    ref: https://github.com//apache/incubator-mxnet/blob/master/python/mxnet/contrib/quantization.py
    ref: https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/_modules/
                 pytorch_quantization/calib/histogram.html
    """

    def __init__(self, method, num_bins, symmetric=True, num_quantized_bins=None, percentile=None):
        self.histogram_dict = {}
        self.method = method
        self.symmetric = symmetric
        self.num_bins = num_bins
        self.num_quantized_bins = num_quantized_bins
        self.percentile = percentile

    def get_histogram_dict(self):
        return self.histogram_dict

    def collect(self, name_to_arr):

        # TODO: Currently we have different collect() for entropy and percentile method respectively.
        #       Need unified collect in the future.
        if self.method == "entropy":
            return self.collect_value(name_to_arr)
        elif self.method == "percentile":
            if self.symmetric:
                return self.collect_absolute_value(name_to_arr)
            else:
                return self.collect_value(name_to_arr)
        else:
            raise ValueError("Only 'entropy' or 'percentile' method are supported")

    def collect_absolute_value(self, name_to_arr):
        """
        Collect histogram on absolute value
        """
        for tensor, data_arr in name_to_arr.items():
            data_arr = np.asarray(data_arr)
            data_arr = data_arr.flatten()
            data_arr = np.absolute(data_arr)  # only consider absolute value
            if tensor not in self.histogram_dict:
                # first time it uses num_bins to compute histogram.
                hist, hist_edges = np.histogram(data_arr, bins=self.num_bins)
                self.histogram_dict[tensor] = (hist, hist_edges)
            else:
                old_histogram = self.histogram_dict[tensor]
                old_hist = old_histogram[0]
                old_hist_edges = old_histogram[1]
                temp_amax = np.max(data_arr)
                if temp_amax > old_hist_edges[-1]:
                    # increase the number of bins
                    width = old_hist_edges[1] - old_hist_edges[0]
                    # NOTE: np.arange may create an extra bin after the one containing temp_amax
                    new_bin_edges = np.arange(old_hist_edges[-1] + width, temp_amax + width, width)
                    old_hist_edges = np.hstack((old_hist_edges, new_bin_edges))
                hist, hist_edges = np.histogram(data_arr, bins=old_hist_edges)
                hist[: len(old_hist)] += old_hist
                self.histogram_dict[tensor] = (hist, hist_edges)

    def collect_value(self, name_to_arr):
        """
        Collect histogram on real value
        """
        for tensor, data_arr in name_to_arr.items():
            data_arr = np.asarray(data_arr)
            data_arr = data_arr.flatten()

            if data_arr.size > 0:
                min_value = np.min(data_arr)
                max_value = np.max(data_arr)
            else:
                min_value = 0
                max_value = 0

            threshold = max(abs(min_value), abs(max_value))
            print('data_arr', data_arr)
            print('tensor in self.histogram_dict', tensor in self.histogram_dict)
            if tensor in self.histogram_dict:
                old_histogram = self.histogram_dict[tensor]
                (old_hist, old_hist_edges, old_min, old_max, old_threshold) = old_histogram
                print('old_threshold', old_threshold)
                print('threshold', threshold)
                self.histogram_dict[tensor] = self.merge_histogram(
                    old_histogram, data_arr, min_value, max_value, threshold
                )
                print('self.histogram_dict[tensor]', self.histogram_dict[tensor])
            else:
                hist, hist_edges = np.histogram(data_arr, self.num_bins, range=(-threshold, threshold))
                self.histogram_dict[tensor] = (
                    hist,
                    hist_edges,
                    min_value,
                    max_value,
                    threshold,
                )

    def merge_histogram(self, old_histogram, data_arr, new_min, new_max, new_threshold):

        (old_hist, old_hist_edges, old_min, old_max, old_threshold) = old_histogram

        if new_threshold <= old_threshold:
            new_hist, _ = np.histogram(data_arr, len(old_hist), range=(-old_threshold, old_threshold))
            return (
                new_hist + old_hist,
                old_hist_edges,
                min(old_min, new_min),
                max(old_max, new_max),
                old_threshold,
            )
        else:
            if old_threshold == 0:
                hist, hist_edges = np.histogram(data_arr, len(old_hist), range=(-new_threshold, new_threshold))
                hist += old_hist
            else:
                old_num_bins = len(old_hist)
                old_stride = 2 * old_threshold / old_num_bins
                half_increased_bins = int((new_threshold - old_threshold) // old_stride + 1)
                new_num_bins = old_num_bins + 2 * half_increased_bins
                new_threshold = half_increased_bins * old_stride + old_threshold
                hist, hist_edges = np.histogram(data_arr, new_num_bins, range=(-new_threshold, new_threshold))
                print('new_num_bins', new_num_bins)
                print('(-new_threshold, new_threshold)', (-new_threshold, new_threshold))
                print('hist', hist, len(hist))
                print('old_hist', old_hist, len(old_hist))
                hist[half_increased_bins : new_num_bins - half_increased_bins] += old_hist
            return (
                hist,
                hist_edges,
                min(old_min, new_min),
                max(old_max, new_max),
                new_threshold,
            )

    def compute_collection_result(self):
        if not self.histogram_dict or len(self.histogram_dict) == 0:
            raise ValueError("Histogram has not been collected. Please run collect() first.")

        if self.method == "entropy":
            return self.compute_entropy()
        elif self.method == "percentile":
            return self.compute_percentile()
        else:
            raise ValueError("Only 'entropy' or 'percentile' method are supported")

    def compute_percentile(self):
        if self.percentile < 0 or self.percentile > 100:
            raise ValueError("Invalid percentile. Must be in range 0 <= percentile <= 100.")

        histogram_dict = self.histogram_dict
        percentile = self.percentile

        thresholds_dict = {}  # per tensor thresholds

        for tensor, histogram in histogram_dict.items():
            hist = histogram[0]
            hist_edges = histogram[1]
            total = hist.sum()
            cdf = np.cumsum(hist / total)
            if self.symmetric:
                idx_right = np.searchsorted(cdf, percentile / 100.0)
                thresholds_dict[tensor] = (
                    -float(hist_edges[idx_right]),
                    float(hist_edges[idx_right]),
                )
            else:
                percent_to_cut_one_side = (100.0 - percentile) / 200.0
                idx_right = np.searchsorted(cdf, 1.0 - percent_to_cut_one_side)
                idx_left = np.searchsorted(cdf, percent_to_cut_one_side)
                thresholds_dict[tensor] = (
                    float(hist_edges[idx_left]),
                    float(hist_edges[idx_right]),
                )

        return thresholds_dict

    def compute_entropy(self):
        histogram_dict = self.histogram_dict
        num_quantized_bins = self.num_quantized_bins

        thresholds_dict = {}  # per tensor thresholds
        # print('histogram_dict', histogram_dict)
        for tensor, histogram in histogram_dict.items():
            optimal_threshold = self.get_entropy_threshold(histogram, num_quantized_bins)
            thresholds_dict[tensor] = optimal_threshold

        return thresholds_dict

    def get_entropy_threshold(self, histogram, num_quantized_bins):
        """Given a dataset, find the optimal threshold for quantizing it.
        The reference distribution is `q`, and the candidate distribution is `p`.
        `q` is a truncated version of the original distribution.
        Ref: http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
        """
        import copy

        from scipy.stats import entropy

        hist = histogram[0]
        hist_edges = histogram[1]
        num_bins = hist.size
        zero_bin_index = num_bins // 2
        num_half_quantized_bin = num_quantized_bins // 2

        kl_divergence = np.zeros(zero_bin_index - num_half_quantized_bin + 1)
        thresholds = [(0, 0) for i in range(kl_divergence.size)]

        # <------------ num bins ---------------->
        #        <--- quantized bins ---->
        # |======|===========|===========|=======|
        #              zero bin index
        #        ^                       ^
        #        |                       |
        #   start index               end index          (start of iteration)
        #     ^                             ^
        #     |                             |
        #  start index                  end index               ...
        # ^                                      ^
        # |                                      |
        # start index                    end index       (end of iteration)

        for i in range(num_half_quantized_bin, zero_bin_index + 1, 1):
            start_index = zero_bin_index - i
            end_index = zero_bin_index + i + 1 if (zero_bin_index + i + 1) <= num_bins else num_bins

            thresholds[i - num_half_quantized_bin] = (
                float(hist_edges[start_index]),
                float(hist_edges[end_index]),
            )

            sliced_distribution = copy.deepcopy(hist[start_index:end_index])

            # reference distribution p
            p = sliced_distribution.copy()  # a copy of np array
            left_outliers_count = sum(hist[:start_index])
            right_outliers_count = sum(hist[end_index:])
            p[0] += left_outliers_count
            p[-1] += right_outliers_count

            # nonzeros[i] incidates whether p[i] is non-zero
            nonzeros = (p != 0).astype(np.int64)

            # quantize p.size bins into quantized bins (default 128 bins)
            quantized_bins = np.zeros(num_quantized_bins, dtype=np.int64)
            num_merged_bins = sliced_distribution.size // num_quantized_bins

            # merge bins into quantized bins
            for index in range(num_quantized_bins):
                start = index * num_merged_bins
                end = start + num_merged_bins
                quantized_bins[index] = sum(sliced_distribution[start:end])
            quantized_bins[-1] += sum(sliced_distribution[num_quantized_bins * num_merged_bins :])

            # in order to compare p and q, we need to make length of q equals to length of p
            # expand quantized bins into p.size bins
            q = np.zeros(p.size, dtype=np.int64)
            for index in range(num_quantized_bins):
                start = index * num_merged_bins
                end = start + num_merged_bins

                norm = sum(nonzeros[start:end])
                if norm != 0:
                    q[start:end] = float(quantized_bins[index]) / float(norm)

            p = smooth_distribution(p)
            q = smooth_distribution(q)

            if isinstance(q, np.ndarray):
                kl_divergence[i - num_half_quantized_bin] = entropy(p, q)
            else:
                kl_divergence[i - num_half_quantized_bin] = float("inf")

        min_kl_divergence_idx = np.argmin(kl_divergence)
        optimal_threshold = thresholds[min_kl_divergence_idx]

        return optimal_threshold

    