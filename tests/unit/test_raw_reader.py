import numpy as np
import pytest
from pydantic import ValidationError

from src.camera.raw_config import RawConfig
from src.camera.raw_reader import RawFormatError, decode_raw, read_raw


def layout(**overrides):
    fields = dict(width=4, height=2, bit_depth=12, storage_format="unpacked16",
                  bayer_pattern="RGGB", black_level=64, white_level=4095)
    return RawConfig(**(fields | overrides))


@pytest.mark.parametrize("bits", [8, 10, 12, 14, 16])
@pytest.mark.parametrize("order", ["little", "big"])
def test_unpacked_16_bit_container(bits, order, tmp_path):
    full_scale = (1 << bits) - 1
    expected = np.array([[0, 2, 15, full_scale], [7, 63, full_scale - 1, 1]], dtype=np.uint16)
    path = tmp_path / "测试.raw"
    encoded = expected.astype("<u2" if order == "little" else ">u2")
    path.write_bytes(b"".join(row.tobytes() + b"\xff" for row in encoded))
    config = layout(bit_depth=bits, endianness=order, white_level=full_scale, row_stride_bytes=9)
    frame = read_raw(path, config)
    assert frame.source_format == "headerless"
    assert frame.mosaic.dtype == np.uint16
    np.testing.assert_array_equal(frame.mosaic, expected)


def test_unpacked8_ignores_stride_padding():
    config = layout(bit_depth=8, white_level=255, storage_format="unpacked8", row_stride_bytes=5)
    assert decode_raw(bytes.fromhex("00 02 7f ff aa 01 80 fe 03 bb"), config).tolist() == [
        [0, 2, 127, 255], [1, 128, 254, 3]
    ]


@pytest.mark.parametrize(("bits", "row_hex", "expected"), [
    (10, "00 55 aa ff e4", [0x000, 0x155, 0x2AA, 0x3FF]),
    (12, "ab 12 3c fe 45 6d", [0xABC, 0x123, 0xFED, 0x456]),
    (14, "04 15 26 ff 41 a9 fe", [0x101, 0x565, 0x9AA, 0x3FFF]),
])
def test_mipi_known_byte_vectors(bits, row_hex, expected):
    row = bytes.fromhex(row_hex)
    config = layout(bit_depth=bits, white_level=(1 << bits) - 1,
                    storage_format="mipi_packed", row_stride_bytes=len(row) + 1)
    payload = row + b"\xee" + bytes(len(row)) + b"\xff"
    assert decode_raw(payload, config).tolist() == [expected, [0] * 4]


@pytest.mark.parametrize("overrides", [
    {"storage_format": "vendor_packed"},
    {"storage_format": "mipi_packed", "bit_depth": 16},
    {"storage_format": "mipi_packed", "bit_depth": 10, "white_level": 1023, "width": 6},
    {"storage_format": "mipi_packed", "endianness": "big"},
    {"row_stride_bytes": 7},
    {"black_level": [64, 64, 4095, 64]},
    {"white_level": 4096},
    {"bayer_pattern": "XTRANS"},
])
def test_rejects_inconsistent_or_unsupported_layout(overrides):
    with pytest.raises(ValidationError):
        layout(**overrides)


@pytest.mark.parametrize("length", [15, 17])
def test_rejects_truncated_and_extra_data(length):
    with pytest.raises(RawFormatError, match="expected 16 bytes"):
        decode_raw(bytes(length), layout())


def test_rejects_invalid_upper_bits():
    with pytest.raises(RawFormatError, match="right-aligned"):
        decode_raw(b"\x00\x10" + bytes(14), layout())


def test_container_errors_name_the_file(tmp_path):
    path = tmp_path / "bad.dng"
    path.write_bytes(b"Not a camera container")
    with pytest.raises(RawFormatError, match="bad.dng"):
        read_raw(path)
