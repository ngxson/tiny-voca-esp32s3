file(GLOB DIZIONE_SRCS "${CMAKE_CURRENT_LIST_DIR}/../../third_party/dizione/*.cpp")
idf_component_register(
    SRCS "main.c" "tv_kernels_s3.c" "kernels_s3.S" "sentences.c" "phonemize.cpp"
         "../boards/${BOARD}/board.c" "../drivers/es8311.c" "../drivers/i2s_player.c" "../drivers/ws2812.c"
         "../../engine/tv.c" "../../engine/tv_dsp.c" ${DIZIONE_SRCS}
    INCLUDE_DIRS "." "../drivers" "../../engine" "../../third_party/dizione"
    REQUIRES esp_partition spi_flash esp_timer esp_driver_usb_serial_jtag esp_driver_gpio esp_driver_i2c esp_driver_i2s esp_driver_rmt)
target_compile_options(${COMPONENT_LIB} PRIVATE -O2 -Wno-unused-parameter -Wno-format-truncation)
set_source_files_properties(${DIZIONE_SRCS} "phonemize.cpp" PROPERTIES COMPILE_OPTIONS "-std=gnu++17;-Wno-error;-Wno-unused-variable;-Wno-missing-field-initializers")
