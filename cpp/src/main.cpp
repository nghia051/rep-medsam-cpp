#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xpad.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor-io/xnpz.hpp>

#include <xtensor/xarray.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xmath.hpp>

#include "lrucache.hpp"

using namespace std::string_literals;
using ImageSize = std::array<size_t, 2>;

constexpr size_t EMBEDDINGS_CACHE_SIZE = 1024;
constexpr size_t IMAGE_ENCODER_INPUT_SIZE = 256;
const ov::Shape INPUT_SHAPE = {1, 3, IMAGE_ENCODER_INPUT_SIZE, IMAGE_ENCODER_INPUT_SIZE};

std::array<size_t, 2> get_preprocess_shape(size_t oldh, size_t oldw)
{
    double scale = 1.0 * IMAGE_ENCODER_INPUT_SIZE / std::max(oldh, oldw);
    size_t newh = scale * oldh + 0.5;
    size_t neww = scale * oldw + 0.5;
    return {newh, neww};
}

template <class T>
T cast_npy_file(xt::detail::npy_file &npy_file)
{
    auto m_typestring = npy_file.m_typestring;
    if (m_typestring == "|u1")
    {
        return npy_file.cast<uint8_t>();
    }
    else if (m_typestring == "<u2")
    {
        return npy_file.cast<uint16_t>();
    }
    else if (m_typestring == "<u4")
    {
        return npy_file.cast<uint32_t>();
    }
    else if (m_typestring == "<u8")
    {
        return npy_file.cast<uint64_t>();
    }
    else if (m_typestring == "|i1")
    {
        return npy_file.cast<int8_t>();
    }
    else if (m_typestring == "<i2")
    {
        return npy_file.cast<int16_t>();
    }
    else if (m_typestring == "<i4")
    {
        return npy_file.cast<int32_t>();
    }
    else if (m_typestring == "<i8")
    {
        return npy_file.cast<int64_t>();
    }
    else if (m_typestring == "<f4")
    {
        return npy_file.cast<float>();
    }
    else if (m_typestring == "<f8")
    {
        return npy_file.cast<double>();
    }
    XTENSOR_THROW(std::runtime_error, "Cast error: unknown format "s + m_typestring);
}

struct Encoder
{
    ov::CompiledModel model;
    ov::InferRequest infer_request;
    ImageSize original_size, new_size;

    Encoder(ov::Core &core, const std::string &model_path)
    {
        model = core.compile_model(model_path, "CPU");
        infer_request = model.create_infer_request();
    }

    void set_sizes(const ImageSize &original_size, const ImageSize &new_size)
    {
        this->original_size = original_size;
        this->new_size = new_size;
    }

    ov::Tensor encode_image(const ov::Tensor &input_tensor)
    {
        infer_request.set_input_tensor(input_tensor);
        infer_request.infer();
        return infer_request.get_output_tensor();
    }

    xt::xtensor<float, 4> preprocess_2D(xt::xtensor<uint8_t, 3> &original_img)
    {
        assert(original_img.shape()[0] == 3);
        cv::Mat mat1(cv::Size(original_size[1], original_size[0]), CV_8UC3, original_img.data()), mat2;
        cv::resize(mat1, mat2, cv::Size(new_size[1], new_size[0]), cv::INTER_LINEAR);

        xt::xtensor<float, 3> img = xt::adapt((uint8_t *)mat2.data, mat2.total() * mat2.channels(), xt::no_ownership(), std::vector<int>{mat2.rows, mat2.cols, mat2.channels()});
        img = (img - xt::amin(img)()) / std::max(xt::amax(img)() - xt::amin(img)(), 1e-8f);

        xt::xtensor<float, 3> padded = xt::pad(img, {{0, IMAGE_ENCODER_INPUT_SIZE - new_size[0]}, {0, IMAGE_ENCODER_INPUT_SIZE - new_size[1]}, {0, 0}});

        xt::xtensor<float, 4> result = xt::zeros<float>(INPUT_SHAPE);

        for (size_t h = 0; h < IMAGE_ENCODER_INPUT_SIZE; ++h)
        {
            for (size_t w = 0; w < IMAGE_ENCODER_INPUT_SIZE; ++w)
            {
                for (size_t c = 0; c < 3; ++c)
                {
                    result(0, c, h, w) = padded(h, w, c);
                }
            }
        }

        return result;
    }
};

struct Decoder
{
    ov::CompiledModel model;
    ov::InferRequest infer_request;
    ImageSize original_size, new_size;

    Decoder(ov::Core &core, const std::string &model_path)
    {
        model = core.compile_model(model_path, "CPU");
        infer_request = model.create_infer_request();
    }

    void set_sizes(const ImageSize &original_size, const ImageSize &new_size)
    {
        this->original_size = original_size;
        this->new_size = new_size;
    }

    void set_embedding_tensor(const ov::Tensor &embedding_tensor)
    {
        infer_request.set_input_tensor(0, embedding_tensor);
    }

    xt::xtensor<float, 4> postprocess_masks(const ov::Tensor &masks)
    {
        auto shape = masks.get_shape();
        size_t B = shape[0]; // Số batch
        size_t H = shape[2]; // Chiều cao ban đầu
        size_t W = shape[3]; // Chiều rộng ban đầu

        const float *data_ptr = static_cast<const float *>(masks.data());

        xt::xtensor<float, 4> new_masks = xt::empty<float>(
            {B, 1ul, original_size[0], original_size[1]});

        // Xử lý từng batch
        for (size_t b = 0; b < B; ++b)
        {
            cv::Mat original_mask(static_cast<int>(H), static_cast<int>(W), CV_32F,
                                  const_cast<float *>(data_ptr + b * H * W));

            // Cắt mask theo new_size (chiều cao = new_size[0], chiều rộng = new_size[1])
            cv::Mat cropped_mask = original_mask(cv::Rect(0, 0, static_cast<int>(new_size[1]), static_cast<int>(new_size[0])));

            // Tạo ma trận đích để chứa mask sau khi thay đổi kích thước
            cv::Mat resized_mask(static_cast<int>(original_size[0]), static_cast<int>(original_size[1]), CV_32F);

            // Thay đổi kích thước về original_size bằng nội suy song tuyến tính
            cv::resize(cropped_mask, resized_mask,
                       cv::Size(static_cast<int>(original_size[1]), static_cast<int>(original_size[0])),
                       0, 0, cv::INTER_LINEAR);

            // Chuyển cv::Mat đã thay đổi kích thước về xtensor và gán vào tensor đầu ra
            auto resized_xt = xt::adapt(reinterpret_cast<float *>(resized_mask.data),
                                        original_size,
                                        xt::layout_type::row_major);
            xt::view(new_masks, b, 0, xt::all(), xt::all()) = resized_xt;
        }

        return new_masks;
    }

    xt::xtensor<uint8_t, 2> decode_mask(const ov::Tensor &boxes_tensor)
    {
        infer_request.set_input_tensor(1, boxes_tensor);
        infer_request.infer();

        ov::Tensor masks_tensor = infer_request.get_output_tensor();

        auto low_res_pred = postprocess_masks(masks_tensor);
        low_res_pred = 1.0 / (1.0 + xt::exp(-low_res_pred));

        xt::xtensor<uint8_t, 4> medsam_seg = xt::where(low_res_pred > 0.5f, 1.0, 0.0);

        xt::xtensor<uint8_t, 2> segs = xt::zeros<uint8_t>({original_size[0], original_size[1]});

        for (int idx = 0; idx < medsam_seg.shape()[0]; ++idx)
        {
            auto mask = xt::view(medsam_seg, idx, 0, xt::all(), xt::all());
            xt::filtration(segs, mask > 0) = idx + 1;
        }

        return segs;
    }
};

void infer_2d(std::string img_file, std::string seg_file, Encoder &encoder, Decoder &decoder)
{
    auto npz_data = xt::load_npz(img_file);
    auto original_img = cast_npy_file<xt::xtensor<uint8_t, 3>>(npz_data["imgs"]);
    auto boxes = cast_npy_file<xt::xtensor<float, 2>>(npz_data["boxes"]);
    assert(original_img.shape()[0] == 3);
    assert(boxes.shape()[1] == 4);

    ImageSize original_size = {original_img.shape()[0], original_img.shape()[1]};
    ImageSize new_size = get_preprocess_shape(original_size[0], original_size[1]);
    float ratio = 1.0f * IMAGE_ENCODER_INPUT_SIZE / std::max(original_size[0], original_size[1]);

    encoder.set_sizes(original_size, new_size);
    decoder.set_sizes(original_size, new_size);

    auto img = encoder.preprocess_2D(original_img);
    ov::Tensor input_tensor(ov::element::f32, INPUT_SHAPE, img.data());

    ov::Tensor embedding_tensor = encoder.encode_image(input_tensor);

    decoder.set_embedding_tensor(embedding_tensor);
    for (size_t i = 0; i < boxes.shape()[0]; ++i)
    {
        for (size_t j = 0; j < boxes.shape()[1]; ++j)
        {
            boxes(i, j) *= ratio;
            boxes(i, j) = int(boxes(i, j));
        }
    }
    ov::Tensor boxes_ov_tensor(
        ov::element::f32,
        {static_cast<size_t>(boxes.shape()[0]), boxes.shape()[1]},
        boxes.data());
    auto segs = decoder.decode_mask(boxes_ov_tensor);
    
    std::remove(seg_file.c_str());
    xt::dump_npz(seg_file, "segs", segs, false);
}

bool starts_with(const std::string &str, const std::string &prefix)
{
    return str.size() >= prefix.size() && str.compare(0, prefix.size(), prefix) == 0;
}

bool ends_with(const std::string &str, const std::string &suffix)
{
    return str.size() >= suffix.size() && str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

int main(int argc, char **argv)
{
    if (argc != 6)
    {
        std::cerr << "Usage: " << argv[0] << " <encoder.xml> <decoder.xml> <model cache folder> <imgs folder> <segs folder>\n";
        return 1;
    }

    ov::Core core;
    core.set_property("CPU", ov::hint::inference_precision(ov::element::f32));
    core.set_property("CPU", ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY));
    core.set_property("CPU", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
    core.set_property("CPU", ov::hint::num_requests(1));
    core.set_property(ov::cache_dir(argv[3]));
    Encoder encoder(core, argv[1]);
    Decoder decoder(core, argv[2]);

    std::filesystem::path imgs_folder(argv[4]);
    if (!std::filesystem::is_directory(imgs_folder))
    {
        throw std::runtime_error(imgs_folder.string() + " is not a folder");
    }

    std::filesystem::path segs_folder(argv[5]);
    if (!std::filesystem::exists(segs_folder) && !std::filesystem::create_directory(segs_folder))
    {
        throw std::runtime_error("Failed to create " + segs_folder.string());
    }
    if (!std::filesystem::is_directory(segs_folder))
    {
        throw std::runtime_error(segs_folder.string() + " is not a folder");
    }

    for (const auto &entry : std::filesystem::directory_iterator(imgs_folder))
    {
        if (!entry.is_regular_file())
        {
            continue;
        }

        auto base_name = entry.path().filename().string();
        if (ends_with(base_name, ".npz"))
        {
            auto img_file = entry.path().string();
            auto seg_file = (segs_folder / entry.path().filename()).string();

            std::cout << "Processing " << base_name << std::endl;
            auto infer_start = std::chrono::high_resolution_clock::now();
            if (starts_with(base_name, "2D"))
            {
                infer_2d(img_file, seg_file, encoder, decoder);
            }
            else
            {
            }
            auto infer_finish = std::chrono::high_resolution_clock::now();
            std::cout << "Inferred " << base_name << " in " << std::chrono::duration_cast<std::chrono::milliseconds>(infer_finish - infer_start).count() << "ms\n";
        }
    }

    return 0;
}
