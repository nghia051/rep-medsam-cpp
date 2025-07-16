```
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release
cmake --build build --verbose -j$(nproc)
```

```
./build/main ../openvino_models/rep_medsam_preprocessed/encoder.xml ../openvino_models/rep_medsam_preprocessed/decoder.xml ../output/rep_medsam_cpp/cache ../dataset/imgs ../output/rep_medsam_cpp/segs
```

```
Total time: 1647071ms

Total time: 363279ms
Total time for processing: 362.7365744114 seconds
```
./build/main ../openvino_models/rep_medsam_preprocessed/encoder.xml ../openvino_models/rep_medsam_preprocessed/decoder.xml ../output/rep_medsam_selected_cpp/cache ../dataset/imgs_selected ../output/rep_medsam_selected_cpp/segs