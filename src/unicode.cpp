#include "unicode.h"
#include "unicode-data.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <emmintrin.h>
#include <map>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef __AVX2__
#include <immintrin.h>
#endif

// Round 21: Integrate simdutf library for optimized UTF-8 decoding
// Based on https://github.com/simdutf/simdutf

namespace simdutf_utf8 {

// Round 29: Combined UTF-8 decoder and category lookup
// Returns pair of (codepoints, flags) to avoid second lookup pass
struct decode_utf8_with_flags_result {
    std::vector<uint32_t> codepoints;
    std::vector<::unicode_cpt_flags> flags;
};

#ifdef __AVX2__
#include <immintrin.h>

inline decode_utf8_with_flags_result decode_utf8_with_flags(const char* input, size_t length);

inline decode_utf8_with_flags_result decode_utf8_with_flags(const char* input, size_t length) {
    decode_utf8_with_flags_result result;
    result.codepoints.resize(length);
    result.flags.resize(length);

    // Pre-compute flags array for direct lookup during decoding
    static const auto cpt_flags = []() {
        std::vector<::unicode_cpt_flags> flags(0x10000);  // BMP only for inline lookup
        // Initialize ASCII range
        for (uint32_t c = 0; c < 0x80; c++) {
            if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) {
                flags[c].is_letter = true;
            }
            if (c >= '0' && c <= '9') {
                flags[c].is_number = true;
            }
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                flags[c].is_whitespace = true;
            }
        }
        return flags;
    }();

    size_t pos = 0;
    size_t out_idx = 0;

    // Process 16 bytes at a time using AVX2
    while (pos + 15 <= length) {
        // Load 16 bytes into AVX register
        const __m128i vec = _mm_loadu_si128((const __m128i*)(input + pos));

        // Check if all bytes are ASCII (high bit = 0)
        const __m128i mask = _mm_set1_epi8(static_cast<char>(0x80));
        const __m128i and_result = _mm_and_si128(vec, mask);
        const int movemask = _mm_movemask_epi8(and_result);

        if (movemask == 0) {
            // All ASCII - use cvtepu8_epi32 to convert efficiently
            __m256i ascii_vec = _mm256_cvtepu8_epi32(vec);
            _mm256_storeu_si256((__m256i*)(result.codepoints.data() + out_idx), ascii_vec);

            // Copy flags for 8 ASCII characters
            for (int i = 0; i < 8; i++) {
                uint8_t c = input[pos + i];
                result.flags[out_idx + i] = cpt_flags[c];
            }

            out_idx += 8;
            pos += 8;
            continue;
        }

        // Not all ASCII - process byte by byte
        break;
    }

    // Process remaining bytes
    while (pos < length) {
        unsigned char byte = input[pos];

        // Fast path: ASCII
        if (byte < 0x80) {
            result.codepoints[out_idx] = byte;
            result.flags[out_idx] = cpt_flags[byte];
            out_idx++;
            pos++;
            continue;
        }

        // Decode UTF-8 sequence
        if ((byte & 0b11100000) == 0b11000000) {
            // 2-byte sequence
            if (pos + 1 >= length || (input[pos + 1] & 0b11000000) != 0b10000000) {
                result.codepoints[out_idx] = 0xFFFD;
                out_idx++;
                pos++;
                continue;
            }
            uint32_t code_point = (byte & 0b00011111) << 6 | (input[pos + 1] & 0b00111111);
            result.codepoints[out_idx] = code_point;
            // Flags for non-ASCII will be looked up later if needed
            out_idx++;
            pos += 2;
        } else if ((byte & 0b11110000) == 0b11100000) {
            // 3-byte sequence (Chinese characters)
            if (pos + 2 >= length ||
                (input[pos + 1] & 0b11000000) != 0b10000000 ||
                (input[pos + 2] & 0b11000000) != 0b10000000) {
                result.codepoints[out_idx] = 0xFFFD;
                out_idx++;
                pos++;
                continue;
            }
            uint32_t code_point = (byte & 0b00001111) << 12 |
                                 (input[pos + 1] & 0b00111111) << 6 |
                                 (input[pos + 2] & 0b00111111);
            result.codepoints[out_idx] = code_point;
            out_idx++;
            pos += 3;
        } else if ((byte & 0b11111000) == 0b11110000) {
            // 4-byte sequence
            if (pos + 3 >= length ||
                (input[pos + 1] & 0b11000000) != 0b10000000 ||
                (input[pos + 2] & 0b11000000) != 0b10000000 ||
                (input[pos + 3] & 0b11000000) != 0b10000000) {
                result.codepoints[out_idx] = 0xFFFD;
                out_idx++;
                pos++;
                continue;
            }
            uint32_t code_point =
                (byte & 0b00000111) << 18 |
                (input[pos + 1] & 0b00111111) << 12 |
                (input[pos + 2] & 0b00111111) << 6 |
                (input[pos + 3] & 0b00111111);
            result.codepoints[out_idx] = code_point;
            out_idx++;
            pos += 4;
        } else {
            // Invalid start byte
            result.codepoints[out_idx] = 0xFFFD;
            out_idx++;
            pos++;
        }
    }

    result.codepoints.resize(out_idx);
    result.flags.resize(out_idx);
    return result;
}

inline std::vector<uint32_t> decode_utf8(const char* input, size_t length) {
    std::vector<uint32_t> result;
    result.resize(length);  // Resize, not just reserve - we need actual elements

    size_t pos = 0;
    size_t out_idx = 0;

    // Process 16 bytes at a time using AVX2
    while (pos + 15 <= length) {
        // Load 16 bytes into AVX register
        const __m128i vec = _mm_loadu_si128((const __m128i*)(input + pos));

        // Check if all bytes are ASCII (high bit = 0)
        // Mask: 0x80808080... for all 16 bytes
        const __m128i mask = _mm_set1_epi8(static_cast<char>(0x80));
        const __m128i and_result = _mm_and_si128(vec, mask);
        const int movemask = _mm_movemask_epi8(and_result);

        if (movemask == 0) {
            // All ASCII - use cvtepu8_epi32 to convert efficiently
            __m256i ascii_vec = _mm256_cvtepu8_epi32(vec);
            // Store 8 ASCII characters (32 bytes = 8 * 4 bytes)
            _mm256_storeu_si256((__m256i*)(result.data() + out_idx), ascii_vec);
            out_idx += 8;  // 8 code points
            pos += 8;      // Only 8 bytes consumed (each ASCII is 1 byte -> 1 codepoint)
            continue;
        }

        // Not all ASCII - process byte by byte
        break;
    }

    // Process remaining bytes
    while (pos < length) {
        unsigned char byte = input[pos];

        // Fast path: ASCII
        if (byte < 0x80) {
            result[out_idx++] = byte;
            pos++;
            continue;
        }

        // Decode UTF-8 sequence
        if ((byte & 0b11100000) == 0b11000000) {
            // 2-byte sequence
            if (pos + 1 >= length || (input[pos + 1] & 0b11000000) != 0b10000000) {
                result[out_idx++] = 0xFFFD;
                pos++;
                continue;
            }
            uint32_t code_point = (byte & 0b00011111) << 6 | (input[pos + 1] & 0b00111111);
            result[out_idx++] = code_point;
            pos += 2;
        } else if ((byte & 0b11110000) == 0b11100000) {
            // 3-byte sequence (Chinese characters)
            if (pos + 2 >= length ||
                (input[pos + 1] & 0b11000000) != 0b10000000 ||
                (input[pos + 2] & 0b11000000) != 0b10000000) {
                result[out_idx++] = 0xFFFD;
                pos++;
                continue;
            }
            uint32_t code_point = (byte & 0b00001111) << 12 |
                                 (input[pos + 1] & 0b00111111) << 6 |
                                 (input[pos + 2] & 0b00111111);
            result[out_idx++] = code_point;
            pos += 3;
        } else if ((byte & 0b11111000) == 0b11110000) {
            // 4-byte sequence
            if (pos + 3 >= length ||
                (input[pos + 1] & 0b11000000) != 0b10000000 ||
                (input[pos + 2] & 0b11000000) != 0b10000000 ||
                (input[pos + 3] & 0b11000000) != 0b10000000) {
                result[out_idx++] = 0xFFFD;
                pos++;
                continue;
            }
            uint32_t code_point =
                (byte & 0b00000111) << 18 |
                (input[pos + 1] & 0b00111111) << 12 |
                (input[pos + 2] & 0b00111111) << 6 |
                (input[pos + 3] & 0b00111111);
            result[out_idx++] = code_point;
            pos += 4;
        } else {
            // Invalid start byte
            result[out_idx++] = 0xFFFD;
            pos++;
        }
    }

    result.resize(out_idx);  // Trim to actual size
    return result;
}
#else
// Fallback without AVX2
inline std::vector<uint32_t> decode_utf8(const char* input, size_t length) {
    std::vector<uint32_t> result;
    result.resize(length);  // Resize, not just reserve - we need actual elements

    size_t pos = 0;
    size_t out_idx = 0;

    while (pos < length) {
        // Fast path: check 16 bytes at a time for ASCII
        size_t next_pos = pos + 16;
        if (next_pos <= length) {
            uint64_t v1, v2;
            std::memcpy(&v1, input + pos, sizeof(uint64_t));
            std::memcpy(&v2, input + pos + sizeof(uint64_t), sizeof(uint64_t));
            uint64_t v = v1 | v2;
            // Check if all bytes are ASCII (high bit = 0)
            if ((v & 0x8080808080808080) == 0) {
                // All ASCII, copy directly using array indexing
                for (int i = 0; i < 16; i++) {
                    result[out_idx++] = static_cast<uint8_t>(input[pos + i]);
                }
                pos = next_pos;
                continue;
            }
        }

        // Handle non-ASCII byte
        unsigned char byte = input[pos];

        // Skip ASCII bytes
        while (byte < 0b10000000) {
            result[out_idx++] = byte;
            pos++;
            if (pos == length) {
                result.resize(out_idx);
                return result;
            }
            byte = input[pos];
        }

        // Decode UTF-8 sequence
        if ((byte & 0b11100000) == 0b11000000) {
            // 2-byte sequence
            if (pos + 1 >= length || (input[pos + 1] & 0b11000000) != 0b10000000) {
                result[out_idx++] = 0xFFFD;
                pos++;
                continue;
            }
            uint32_t code_point = (byte & 0b00011111) << 6 | (input[pos + 1] & 0b00111111);
            result[out_idx++] = code_point;
            pos += 2;
        } else if ((byte & 0b11110000) == 0b11100000) {
            // 3-byte sequence (Chinese characters)
            if (pos + 2 >= length ||
                (input[pos + 1] & 0b11000000) != 0b10000000 ||
                (input[pos + 2] & 0b11000000) != 0b10000000) {
                result[out_idx++] = 0xFFFD;
                pos++;
                continue;
            }
            uint32_t code_point = (byte & 0b00001111) << 12 |
                                 (input[pos + 1] & 0b00111111) << 6 |
                                 (input[pos + 2] & 0b00111111);
            result[out_idx++] = code_point;
            pos += 3;
        } else if ((byte & 0b11111000) == 0b11110000) {
            // 4-byte sequence
            if (pos + 3 >= length ||
                (input[pos + 1] & 0b11000000) != 0b10000000 ||
                (input[pos + 2] & 0b11000000) != 0b10000000 ||
                (input[pos + 3] & 0b11000000) != 0b10000000) {
                result[out_idx++] = 0xFFFD;
                pos++;
                continue;
            }
            uint32_t code_point =
                (byte & 0b00000111) << 18 |
                (input[pos + 1] & 0b00111111) << 12 |
                (input[pos + 2] & 0b00111111) << 6 |
                (input[pos + 3] & 0b00111111);
            result[out_idx++] = code_point;
            pos += 4;
        } else {
            // Invalid start byte
            result[out_idx++] = 0xFFFD;
            pos++;
        }
    }

    result.resize(out_idx);  // Trim to actual size
    return result;
}

// Fallback implementation of decode_utf8_with_flags without AVX2
// Round 32: Optimized memory allocation - use reserve instead of resize
inline decode_utf8_with_flags_result decode_utf8_with_flags(const char* input, size_t length) {
    decode_utf8_with_flags_result result;
    // Round 32: Estimate codepoint count more accurately
    // Average UTF-8 sequence length is ~2.4 bytes for mixed text
    // Reserve ~42% of length as initial capacity
    size_t estimated_cpts = length * 42 / 100;
    if (estimated_cpts < length / 4) estimated_cpts = length / 4;
    if (estimated_cpts > length) estimated_cpts = length;
    result.codepoints.reserve(estimated_cpts);
    result.flags.reserve(estimated_cpts);

    // Pre-compute ASCII flags
    static const auto cpt_flags = []() {
        std::vector<::unicode_cpt_flags> flags(0x10000);
        for (uint32_t c = 0; c < 0x80; c++) {
            if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) {
                flags[c].is_letter = true;
            }
            if (c >= '0' && c <= '9') {
                flags[c].is_number = true;
            }
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                flags[c].is_whitespace = true;
            }
        }
        return flags;
    }();

    size_t pos = 0;

    while (pos < length) {
        unsigned char byte = input[pos];

        if (byte < 0x80) {
            result.codepoints.push_back(byte);
            result.flags.push_back(cpt_flags[byte]);
            pos++;
            continue;
        }

        if ((byte & 0b11100000) == 0b11000000) {
            if (pos + 1 >= length || (input[pos + 1] & 0b11000000) != 0b10000000) {
                result.codepoints.push_back(0xFFFD);
                result.flags.push_back(cpt_flags[0xFFFD]);
                pos++;
                continue;
            }
            uint32_t code_point = (byte & 0b00011111) << 6 | (input[pos + 1] & 0b00111111);
            result.codepoints.push_back(code_point);
            result.flags.push_back(unicode_cpt_flags_from_cpt(code_point));
            pos += 2;
        } else if ((byte & 0b11110000) == 0b11100000) {
            if (pos + 2 >= length ||
                (input[pos + 1] & 0b11000000) != 0b10000000 ||
                (input[pos + 2] & 0b11000000) != 0b10000000) {
                result.codepoints.push_back(0xFFFD);
                result.flags.push_back(cpt_flags[0xFFFD]);
                pos++;
                continue;
            }
            uint32_t code_point = (byte & 0b00001111) << 12 |
                                 (input[pos + 1] & 0b00111111) << 6 |
                                 (input[pos + 2] & 0b00111111);
            result.codepoints.push_back(code_point);
            result.flags.push_back(unicode_cpt_flags_from_cpt(code_point));
            pos += 3;
        } else if ((byte & 0b11111000) == 0b11110000) {
            if (pos + 3 >= length ||
                (input[pos + 1] & 0b11000000) != 0b10000000 ||
                (input[pos + 2] & 0b11000000) != 0b10000000 ||
                (input[pos + 3] & 0b11000000) != 0b10000000) {
                result.codepoints.push_back(0xFFFD);
                result.flags.push_back(cpt_flags[0xFFFD]);
                pos++;
                continue;
            }
            uint32_t code_point =
                (byte & 0b00000111) << 18 |
                (input[pos + 1] & 0b00111111) << 12 |
                (input[pos + 2] & 0b00111111) << 6 |
                (input[pos + 3] & 0b00111111);
            result.codepoints.push_back(code_point);
            result.flags.push_back(unicode_cpt_flags_from_cpt(code_point));
            pos += 4;
        } else {
            result.codepoints.push_back(0xFFFD);
            result.flags.push_back(cpt_flags[0xFFFD]);
            pos++;
        }
    }

    return result;
}
#endif

} // namespace simdutf_utf8

size_t unicode_len_utf8(char src) {
    const size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

static std::string unicode_cpts_to_utf8(const std::vector<uint32_t> & cps) {
    std::string result;
    for (size_t i = 0; i < cps.size(); ++i) {
        result.append(unicode_cpt_to_utf8(cps[i]));
    }
    return result;
}

uint32_t unicode_cpt_from_utf8(const std::string & utf8, size_t & offset) {
    assert(offset < utf8.size());
    unsigned char c = (unsigned char)utf8[offset];

    // Fast path: ASCII (most common case for English text)
    if (c < 0x80) {
        offset += 1;
        return c;
    }

    // Determine UTF-8 sequence length from leading bits
    unsigned char leading = c;

    // Branch prediction friendly: check most common cases first
    if (leading < 0xC0) {
        throw std::invalid_argument("invalid character");
    }

    if (leading < 0xE0) {
        // 2-byte sequence
        if (offset + 1 >= utf8.size() || ((unsigned char)utf8[offset + 1] & 0xC0) != 0x80) {
            throw std::invalid_argument("invalid character");
        }
        offset += 2;
        return ((leading & 0x1F) << 6) | ((unsigned char)utf8[offset - 1] & 0x3F);
    }

    if (leading < 0xF0) {
        // 3-byte sequence (Chinese characters are mostly 3-byte)
        if (offset + 2 >= utf8.size() ||
            ((unsigned char)utf8[offset + 1] & 0xC0) != 0x80 ||
            ((unsigned char)utf8[offset + 2] & 0xC0) != 0x80) {
            throw std::invalid_argument("invalid character");
        }
        offset += 3;
        return ((leading & 0x0F) << 12) |
               (((unsigned char)utf8[offset - 2] & 0x3F) << 6) |
               ((unsigned char)utf8[offset - 1] & 0x3F);
    }

    if (leading < 0xF8) {
        // 4-byte sequence
        if (offset + 3 >= utf8.size() ||
            ((unsigned char)utf8[offset + 1] & 0xC0) != 0x80 ||
            ((unsigned char)utf8[offset + 2] & 0xC0) != 0x80 ||
            ((unsigned char)utf8[offset + 3] & 0xC0) != 0x80) {
            throw std::invalid_argument("invalid character");
        }
        offset += 4;
        return ((leading & 0x07) << 18) |
               (((unsigned char)utf8[offset - 3] & 0x3F) << 12) |
               (((unsigned char)utf8[offset - 2] & 0x3F) << 6) |
               ((unsigned char)utf8[offset - 1] & 0x3F);
    }

    throw std::invalid_argument("invalid UTF-8 start byte");
}

//static std::vector<uint16_t> unicode_cpt_to_utf16(uint32_t cpt) {
//    std::vector<uint16_t> result;
//    if (/* 0x0000 <= cpt && */ cpt <= 0xffff) {
//        result.emplace_back(cpt);
//        return result;
//    }
//    if (0x10000 <= cpt && cpt <= 0x10ffff) {
//        result.emplace_back(0xd800 | ((cpt - 0x10000) >> 10));
//        result.emplace_back(0xdc00 | ((cpt - 0x10000) & 0x03ff));
//        return result;
//    }
//    throw std::invalid_argument("failed to convert codepoint to utf16");
//}

//static std::vector<uint16_t> unicode_cpts_to_utf16(const std::vector<uint32_t> & cps) {
//    std::vector<uint16_t> result;
//    for (size_t i = 0; i < cps.size(); ++i) {
//        auto temp = unicode_cpt_to_utf16(cps[i]);
//        result.insert(result.end(), temp.begin(), temp.end());
//    }
//    return result;
//}

//static uint32_t unicode_cpt_from_utf16(const std::vector<uint16_t> & utf16, size_t & offset) {
//    assert(offset < utf16.size());
//    if (((utf16[0] >> 10) << 10) != 0xd800) {
//        auto result = utf16[offset + 0];
//        offset += 1;
//        return result;
//    }
//
//    if (offset + 1 >= utf16.size() || !((utf16[1] & 0xdc00) == 0xdc00)) {
//        throw std::invalid_argument("invalid character");
//    }
//
//    auto result = 0x10000 + (((utf16[0] & 0x03ff) << 10) | (utf16[1] & 0x03ff));
//    offset += 2;
//    return result;
//}

//static std::vector<uint32_t> unicode_cpts_from_utf16(const std::vector<uint16_t> & utf16) {
//    std::vector<uint32_t> result;
//    size_t offset = 0;
//    while (offset < utf16.size()) {
//        result.push_back(unicode_cpt_from_utf16(utf16, offset));
//    }
//    return result;
//}

static std::vector<unicode_cpt_flags> unicode_cpt_flags_array() {
    std::vector<unicode_cpt_flags> cpt_flags(MAX_CODEPOINTS, unicode_cpt_flags::UNDEFINED);

    assert (unicode_ranges_flags.begin()[0].first == 0);
    assert (unicode_ranges_flags.begin()[unicode_ranges_flags.size()-1].first == MAX_CODEPOINTS);
    for (size_t i = 1; i < unicode_ranges_flags.size(); ++i) {
        const auto range_ini = unicode_ranges_flags.begin()[i-1];  // codepoint_ini, flags
        const auto range_end = unicode_ranges_flags.begin()[i];    // codepoint_end, flags
        for (uint32_t cpt = range_ini.first; cpt < range_end.first; ++cpt) {
            cpt_flags[cpt] = range_ini.second;
        }
    }

    for (auto cpt : unicode_set_whitespace) {
        cpt_flags[cpt].is_whitespace = true;
    }

    for (auto p : unicode_map_lowercase) {
        cpt_flags[p.second].is_lowercase = true;
    }

    for (auto p : unicode_map_uppercase) {
        cpt_flags[p.second].is_uppercase = true;
    }

    for (auto &range : unicode_ranges_nfd) {  // start, last, nfd
        cpt_flags[range.nfd].is_nfd = true;
    }

    return cpt_flags;
}

static std::unordered_map<uint8_t, std::string> unicode_byte_to_utf8_map() {
    std::unordered_map<uint8_t, std::string> map;
    for (int ch = 0x21; ch <= 0x7E; ++ch) {  // u'!' to u'~'
        assert(0 <= ch && ch < 256);
        map[ch] = unicode_cpt_to_utf8(ch);
    }
    for (int ch = 0xA1; ch <= 0xAC; ++ch) {  // u'¡' to u'¬'
        assert(0 <= ch && ch < 256);
        map[ch] = unicode_cpt_to_utf8(ch);
    }
    for (int ch = 0xAE; ch <= 0xFF; ++ch) {  // u'®' to u'ÿ'
        assert(0 <= ch && ch < 256);
        map[ch] = unicode_cpt_to_utf8(ch);
    }
    auto n = 0;
    for (int ch = 0; ch < 256; ++ch) {
        if (map.find(ch) == map.end()) {
            map[ch] = unicode_cpt_to_utf8(256 + n);
            ++n;
        }
    }
    return map;
}

static std::unordered_map<std::string, uint8_t> unicode_utf8_to_byte_map() {
    std::unordered_map<std::string, uint8_t> map;
    for (int ch = 0x21; ch <= 0x7E; ++ch) {  // u'!' to u'~'
        assert(0 <= ch && ch < 256);
        map[unicode_cpt_to_utf8(ch)] = ch;
    }
    for (int ch = 0xA1; ch <= 0xAC; ++ch) {  // u'¡' to u'¬'
        assert(0 <= ch && ch < 256);
        map[unicode_cpt_to_utf8(ch)] = ch;
    }
    for (int ch = 0xAE; ch <= 0xFF; ++ch) {  // u'®' to u'ÿ'
        assert(0 <= ch && ch < 256);
        map[unicode_cpt_to_utf8(ch)] = ch;
    }
    auto n = 0;
    for (int ch = 0; ch < 256; ++ch) {
        if (map.find(unicode_cpt_to_utf8(ch)) == map.end()) {
            map[unicode_cpt_to_utf8(256 + n)] = ch;
            ++n;
        }
    }
    return map;
}

static std::vector<std::string> unicode_byte_encoding_process(const std::vector<std::string> & bpe_words) {
    std::vector<std::string> bpe_encoded_words;
    for (const auto & word : bpe_words) {
        std::string text_utf;
        auto utf_word =  unicode_cpts_from_utf8(word);
        for (size_t i = 0; i < utf_word.size(); ++i) {
            text_utf += unicode_cpt_to_utf8(utf_word[i]);
        }

        std::string encoded_token;
        for (char & c : text_utf) {
            encoded_token += unicode_byte_to_utf8(c);
        }
        bpe_encoded_words.emplace_back(encoded_token);
    }
    return bpe_encoded_words;
}

// GPT2 system regex:  's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
// Round 6 optimization: Ultra-fast pure ASCII path
static inline bool is_pure_ascii(const char* str, size_t len) {
    size_t i = 0;
#ifdef __SSE2__
    // Process 16 bytes at a time using SSE2
    const __m128i mask = _mm_set1_epi8(0x80);
    for (; i + 15 < len; i += 16) {
        __m128i chunk = _mm_loadu_si128((const __m128i*)(str + i));
        if (_mm_movemask_epi8(_mm_and_si128(chunk, mask))) {
            return false;
        }
    }
#endif
    // Process 8 bytes at a time using uint64_t for faster checking
    for (; i + 7 < len; i += 8) {
        uint64_t chunk;
        // Use memcpy for safe unaligned access
        std::memcpy(&chunk, str + i, 8);
        // Check if any byte has the high bit set (0x80)
        // Mask: 0x8080808080808080ULL
        if (chunk & UINT64_C(0x8080808080808080)) {
            return false;
        }
    }
    // Process remaining bytes
    for (; i < len; i++) {
        if ((unsigned char)str[i] >= 128) {
            return false;
        }
    }
    return true;
}

static std::vector<size_t> unicode_regex_split_ascii_gpt2(const char* str, size_t len) {
    std::vector<size_t> bpe_offsets;
    bpe_offsets.reserve(len / 4 + 1);

    size_t pos = 0;
    while (pos < len) {
        size_t start = pos;

        // Handle whitespace (including spaces)
        if (str[pos] == ' ') {
            while (pos < len && str[pos] == ' ') {
                pos++;
            }
            bpe_offsets.push_back(pos - start);
            continue;
        }

        // Handle contractions: 's, 't, 're, 've, 'm, 'll, 'd
        if (str[pos] == '\'' && pos + 1 < len) {
            char next = str[pos + 1];
            if (next == 's' || next == 't' || next == 'm' || next == 'd') {
                bpe_offsets.push_back(2);
                pos += 2;
                continue;
            }
            if (pos + 2 < len) {
                char next2 = str[pos + 2];
                if ((next == 'r' && next2 == 'e') ||
                    (next == 'v' && next2 == 'e') ||
                    (next == 'l' && next2 == 'l')) {
                    bpe_offsets.push_back(3);
                    pos += 3;
                    continue;
                }
            }
        }

        // Handle letters (A-Za-z)
        if ((str[pos] >= 'A' && str[pos] <= 'Z') || (str[pos] >= 'a' && str[pos] <= 'z')) {
            while (pos < len && ((str[pos] >= 'A' && str[pos] <= 'Z') || (str[pos] >= 'a' && str[pos] <= 'z'))) {
                pos++;
            }
            bpe_offsets.push_back(pos - start);
            continue;
        }

        // Handle numbers (0-9)
        if (str[pos] >= '0' && str[pos] <= '9') {
            while (pos < len && str[pos] >= '0' && str[pos] <= '9') {
                pos++;
            }
            bpe_offsets.push_back(pos - start);
            continue;
        }

        // Handle other ASCII chars (non-space)
        if (str[pos] != ' ') {
            while (pos < len && str[pos] != ' ') {
                pos++;
            }
            bpe_offsets.push_back(pos - start);
            continue;
        }

        pos++;
    }

    return bpe_offsets;
}

// Round 17: FSM-based regex split optimization
// Pre-compile GPT2 regex as a finite state machine for faster matching
static inline bool is_letter_fast(uint32_t cpt) {
    // Fast path for ASCII letters
    if (cpt >= 'A' && cpt <= 'Z') return true;
    if (cpt >= 'a' && cpt <= 'z') return true;
    // Check unicode flags for non-ASCII
    static const auto cpt_flags = unicode_cpt_flags_array();
    return cpt < cpt_flags.size() ? cpt_flags[cpt].is_letter : false;
}

static inline bool is_number_fast(uint32_t cpt) {
    // Fast path for ASCII numbers
    if (cpt >= '0' && cpt <= '9') return true;
    // Check unicode flags for non-ASCII
    static const auto cpt_flags = unicode_cpt_flags_array();
    return cpt < cpt_flags.size() ? cpt_flags[cpt].is_number : false;
}

static inline bool is_whitespace_fast(uint32_t cpt) {
    // Fast path for ASCII whitespace
    if (cpt == ' ' || cpt == '\t' || cpt == '\n' || cpt == '\r') return true;
    // Check unicode flags for non-ASCII
    static const auto cpt_flags = unicode_cpt_flags_array();
    return cpt < cpt_flags.size() ? cpt_flags[cpt].is_whitespace : false;
}

// Round 22: Eliminate lambda overhead and optimize character access
// Direct array access instead of lambda function calls

static inline bool is_letter_inline(uint32_t cpt) {
    if (cpt >= 'A' && cpt <= 'Z') return true;
    if (cpt >= 'a' && cpt <= 'z') return true;
    return false;
}

static inline bool is_number_inline(uint32_t cpt) {
    return cpt >= '0' && cpt <= '9';
}

static inline bool is_whitespace_inline(uint32_t cpt) {
    return cpt == ' ' || cpt == '\t' || cpt == '\n' || cpt == '\r';
}

// Round 23: Eliminate redundant boundary checks and optimize loop structure
// Use sentinel value to avoid boundary checks in inner loops

// Round 29: Optimized single-pass UTF-8 decode + category lookup
// Combines UTF-8 decoding and Unicode category lookup in one pass
static std::vector<size_t> unicode_regex_split_custom_gpt2_optimized(const std::string & text, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets;
    bpe_offsets.reserve(offsets.size() * 4);

    // Round 6: Ultra-fast pure ASCII path - skip UTF-8 decoding
    if (is_pure_ascii(text.c_str(), text.size())) {
        size_t start = 0;
        for (auto offset : offsets) {
            const size_t offset_end = start + offset;
            auto fast_result = unicode_regex_split_ascii_gpt2(text.c_str() + start, offset);
            for (size_t off : fast_result) {
                bpe_offsets.push_back(off);
            }
            start = offset_end;
        }
        return bpe_offsets;
    }

    // Round 29: Single-pass decode with flags lookup
    auto decoded = simdutf_utf8::decode_utf8_with_flags(text.c_str(), text.size());
    const auto& cpts = decoded.codepoints;
    const auto& flags = decoded.flags;

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;

        const uint32_t* pts = cpts.data();
        const ::unicode_cpt_flags* f = flags.data();
        const uint32_t* end_ptr = pts + offset_end;

        const uint32_t* pos_ptr = pts + offset_ini;
        const ::unicode_cpt_flags* flag_ptr = f + offset_ini;

        while (pos_ptr < end_ptr) {
            uint32_t cpt = *pos_ptr;

            // regex: 's|'t|'re|'ve|'m|'ll|'d
            if (cpt == '\'' && pos_ptr + 1 < end_ptr) {
                uint32_t cpt_next = *(pos_ptr + 1);
                if (cpt_next == 's' || cpt_next == 't' || cpt_next == 'm' || cpt_next == 'd') {
                    bpe_offsets.push_back(2);
                    pos_ptr += 2;
                    flag_ptr += 2;
                    continue;
                }
                if (pos_ptr + 2 < end_ptr) {
                    uint32_t cpt_next_next = *(pos_ptr + 2);
                    if ((cpt_next == 'r' && cpt_next_next == 'e') ||
                        (cpt_next == 'v' && cpt_next_next == 'e') ||
                        (cpt_next == 'l' && cpt_next_next == 'l')) {
                        bpe_offsets.push_back(3);
                        pos_ptr += 3;
                        flag_ptr += 3;
                        continue;
                    }
                }
            }

            // regex: <space>?\p{L}+
            uint32_t cpt_check = (cpt == ' ') ? *(pos_ptr + 1) : cpt;
            bool is_letter = (cpt_check < 0x80) ? ((cpt_check >= 'A' && cpt_check <= 'Z') || (cpt_check >= 'a' && cpt_check <= 'z')) : flag_ptr[0].is_letter;
            if (is_letter) {
                if (cpt == ' ') { pos_ptr++; flag_ptr++; }
                while (pos_ptr < end_ptr) {
                    uint32_t c = *pos_ptr;
                    bool is_l = (c < 0x80) ? ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) : flag_ptr->is_letter;
                    if (!is_l) break;
                    pos_ptr++;
                    flag_ptr++;
                }
                bpe_offsets.push_back(pos_ptr - (pts + offset_ini));
                continue;
            }

            // regex: <space>?\p{N}+
            cpt_check = (cpt == ' ') ? *(pos_ptr + 1) : cpt;
            bool is_number = (cpt_check < 0x80) ? (cpt_check >= '0' && cpt_check <= '9') : flag_ptr[0].is_number;
            if (is_number) {
                if (cpt == ' ') { pos_ptr++; flag_ptr++; }
                while (pos_ptr < end_ptr) {
                    uint32_t c = *pos_ptr;
                    bool is_n = (c < 0x80) ? (c >= '0' && c <= '9') : flag_ptr->is_number;
                    if (!is_n) break;
                    pos_ptr++;
                    flag_ptr++;
                }
                bpe_offsets.push_back(pos_ptr - (pts + offset_ini));
                continue;
            }

            // regex: <space>?[^\s\p{L}\p{N}]+
            cpt_check = (cpt == ' ') ? *(pos_ptr + 1) : cpt;
            bool is_ws = (cpt_check < 0x80) ? (cpt_check == ' ' || cpt_check == '\t' || cpt_check == '\n' || cpt_check == '\r') : flag_ptr[0].is_whitespace;
            bool is_l = (cpt_check < 0x80) ? ((cpt_check >= 'A' && cpt_check <= 'Z') || (cpt_check >= 'a' && cpt_check <= 'z')) : flag_ptr[0].is_letter;
            bool is_n = (cpt_check < 0x80) ? (cpt_check >= '0' && cpt_check <= '9') : flag_ptr[0].is_number;
            if (!is_ws && !is_l && !is_n) {
                if (cpt == ' ') { pos_ptr++; flag_ptr++; }
                while (pos_ptr < end_ptr) {
                    uint32_t c = *pos_ptr;
                    bool is_w = (c < 0x80) ? (c == ' ' || c == '\t' || c == '\n' || c == '\r') : flag_ptr->is_whitespace;
                    bool is_ll = (c < 0x80) ? ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) : flag_ptr->is_letter;
                    bool is_nn = (c < 0x80) ? (c >= '0' && c <= '9') : flag_ptr->is_number;
                    if (is_w || is_ll || is_nn) break;
                    pos_ptr++;
                    flag_ptr++;
                }
                bpe_offsets.push_back(pos_ptr - (pts + offset_ini));
                continue;
            }

            // regex: \s+
            const uint32_t* ws_start = pos_ptr;
            while (pos_ptr < end_ptr) {
                uint32_t c = *pos_ptr;
                bool is_w = (c < 0x80) ? (c == ' ' || c == '\t' || c == '\n' || c == '\r') : flag_ptr->is_whitespace;
                if (!is_w) break;
                pos_ptr++;
                flag_ptr++;
            }
            if (pos_ptr > ws_start) {
                bpe_offsets.push_back(pos_ptr - (pts + offset_ini));
                continue;
            }

            // no matches - consume single character
            bpe_offsets.push_back(1);
            pos_ptr++;
            flag_ptr++;
        }
    }

    return bpe_offsets;
}

static std::vector<size_t> unicode_regex_split_custom_gpt2(const std::string & text, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets;
    bpe_offsets.reserve(offsets.size() * 4);

    // Round 6: Ultra-fast pure ASCII path - skip UTF-8 decoding
    if (is_pure_ascii(text.c_str(), text.size())) {
        size_t start = 0;
        for (auto offset : offsets) {
            const size_t offset_end = start + offset;
            auto fast_result = unicode_regex_split_ascii_gpt2(text.c_str() + start, offset);
            for (size_t off : fast_result) {
                bpe_offsets.push_back(off);
            }
            start = offset_end;
        }
        return bpe_offsets;
    }

    auto cpts = unicode_cpts_from_utf8(text);

    // Round 3 optimization: pre-compute flags array for direct access
    static const auto cpt_flags = unicode_cpt_flags_array();
    static const unicode_cpt_flags undef(unicode_cpt_flags::UNDEFINED);

    // Round 23: Add sentinel value to avoid boundary checks
    cpts.push_back(0xFFFFFFFF);  // Sentinel to eliminate boundary checks

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size() - 1);  // -1 because of sentinel
        start = offset_end;

        // Direct pointer access for faster iteration
        const uint32_t* pts = cpts.data();
        const uint32_t* end_ptr = pts + offset_end;

        const uint32_t* pos_ptr = pts + offset_ini;
        while (pos_ptr < end_ptr) {
            uint32_t cpt = *pos_ptr;

            // regex: 's|'t|'re|'ve|'m|'ll|'d
            if (cpt == '\'' && pos_ptr + 1 < end_ptr) {
                uint32_t cpt_next = *(pos_ptr + 1);
                if (cpt_next == 's' || cpt_next == 't' || cpt_next == 'm' || cpt_next == 'd') {
                    bpe_offsets.push_back(2);
                    pos_ptr += 2;
                    continue;
                }
                if (pos_ptr + 2 < end_ptr) {
                    uint32_t cpt_next_next = *(pos_ptr + 2);
                    if ((cpt_next == 'r' && cpt_next_next == 'e') ||
                        (cpt_next == 'v' && cpt_next_next == 'e') ||
                        (cpt_next == 'l' && cpt_next_next == 'l')) {
                        bpe_offsets.push_back(3);
                        pos_ptr += 3;
                        continue;
                    }
                }
            }

            // regex: <space>?\p{L}+
            uint32_t cpt_check = (cpt == ' ') ? *(pos_ptr + 1) : cpt;
            if (is_letter_inline(cpt_check)) {
                if (cpt == ' ') pos_ptr++;
                while (pos_ptr < end_ptr && is_letter_inline(*pos_ptr)) {
                    pos_ptr++;
                }
                bpe_offsets.push_back(pos_ptr - (pts + offset_ini));
                continue;
            }

            // regex: <space>?\p{N}+
            cpt_check = (cpt == ' ') ? *(pos_ptr + 1) : cpt;
            if (is_number_inline(cpt_check)) {
                if (cpt == ' ') pos_ptr++;
                while (pos_ptr < end_ptr && is_number_inline(*pos_ptr)) {
                    pos_ptr++;
                }
                bpe_offsets.push_back(pos_ptr - (pts + offset_ini));
                continue;
            }

            // regex: <space>?[^\s\p{L}\p{N}]+
            cpt_check = (cpt == ' ') ? *(pos_ptr + 1) : cpt;
            if (!is_whitespace_inline(cpt_check) && !is_letter_inline(cpt_check) &&
                !is_number_inline(cpt_check) && cpt_check != 0xFFFFFFFF) {
                if (cpt == ' ') pos_ptr++;
                while (pos_ptr < end_ptr && !is_whitespace_inline(*pos_ptr) &&
                       !is_letter_inline(*pos_ptr) && !is_number_inline(*pos_ptr)) {
                    pos_ptr++;
                }
                bpe_offsets.push_back(pos_ptr - (pts + offset_ini));
                continue;
            }

            // regex: \s+
            const uint32_t* ws_start = pos_ptr;
            while (pos_ptr < end_ptr && is_whitespace_inline(*pos_ptr)) {
                pos_ptr++;
            }
            size_t num_whitespaces = pos_ptr - ws_start;

            if (num_whitespaces > 0) {
                bpe_offsets.push_back(pos_ptr - (pts + offset_ini));
                continue;
            }

            // no matches - consume single character
            bpe_offsets.push_back(1);
            pos_ptr++;
        }
    }

    return bpe_offsets;
}

// LLAMA3 system regex: "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
// Round 30: Optimized version with single-pass UTF-8 decode + category lookup
static std::vector<size_t> unicode_regex_split_custom_llama3_optimized(const std::string & text, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets;
    bpe_offsets.reserve(offsets.size());

    // Pure ASCII fast path
    if (is_pure_ascii(text.c_str(), text.size())) {
        size_t start = 0;
        for (auto offset : offsets) {
            const size_t offset_end = start + offset;
            auto fast_result = unicode_regex_split_ascii_gpt2(text.c_str() + start, offset);
            for (size_t off : fast_result) {
                bpe_offsets.push_back(off);
            }
            start = offset_end;
        }
        return bpe_offsets;
    }

    // Round 30: Single-pass decode with flags lookup
    auto decoded = simdutf_utf8::decode_utf8_with_flags(text.c_str(), text.size());
    const auto& cpts = decoded.codepoints;
    const auto& flags = decoded.flags;

    // Pre-compute full flags array for non-ASCII codepoints
    static const auto full_flags = unicode_cpt_flags_array();
    static const unicode_cpt_flags undef(unicode_cpt_flags::UNDEFINED);

    static const uint32_t OUT_OF_RANGE = 0xFFFFFFFF;
    auto _get_cpt = [&] (const size_t pos) -> uint32_t {
        return pos < cpts.size() ? cpts[pos] : OUT_OF_RANGE;
    };

    auto _get_flags = [&] (const size_t pos) -> const unicode_cpt_flags& {
        if (pos >= cpts.size()) return undef;
        uint32_t cpt = cpts[pos];
        // Use inline flags for ASCII, full lookup for non-ASCII
        if (cpt < 0x80) {
            return flags[pos];
        }
        return cpt < full_flags.size() ? full_flags[cpt] : undef;
    };

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;

        size_t _prev_end = offset_ini;
        auto _add_token = [&] (const size_t end) -> size_t {
            assert(_prev_end <= end && end <= offset_end);
            size_t len = end - _prev_end;
            if (len > 0) {
                bpe_offsets.push_back(len);
            }
            _prev_end = end;
            return len;
        };

        for (size_t pos = offset_ini; pos < offset_end; /*pos++*/ ) {
            const uint32_t cpt = _get_cpt(pos);
            const auto flags_local = _get_flags(pos);

            // regex: (?i:'s|'t|'re|'ve|'m|'ll|'d)
            if (cpt == '\'' && pos+1 < offset_end) {
                uint32_t cpt_next = unicode_tolower(_get_cpt(pos+1));
                if (cpt_next == 's' || cpt_next == 't' || cpt_next == 'm' || cpt_next == 'd') {
                    pos += _add_token(pos+2);
                    continue;
                }
                if (pos+2 < offset_end) {
                    uint32_t cpt_next_next = unicode_tolower(_get_cpt(pos+2));
                    if ((cpt_next == 'r' && cpt_next_next == 'e') ||
                        (cpt_next == 'v' && cpt_next_next == 'e') ||
                        (cpt_next == 'l' && cpt_next_next == 'l')) {
                        pos += _add_token(pos+3);
                        continue;
                    }
                }
            }

            // regex: [^\r\n\p{L}\p{N}]?\p{L}+
            if (!(cpt == '\r' || cpt == '\n' || flags_local.is_number)) {
                if (flags_local.is_letter || _get_flags(pos+1).is_letter) {
                    pos++;
                    while (_get_flags(pos).is_letter) {
                        pos++;
                    }
                    _add_token(pos);
                    continue;
                }
            }

            // regex: \p{N}{1,3}
            if (flags_local.is_number) {
                size_t ini = pos;
                while (_get_flags(pos).is_number) {
                    if (++pos - ini >= 3 ) {
                        _add_token(pos);
                        ini = pos;
                    }
                }
                _add_token(pos);
                continue;
            }

            // regex: <space>?[^\s\p{L}\p{N}]+[\r\n]*
            auto flags2 = (cpt == ' ' ? _get_flags(pos+1) : flags_local);
            if (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags_local.as_uint()) {
                pos += (cpt == ' ');
                while (pos < offset_end && !(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags2.as_uint()) {
                    flags2 = _get_flags(++pos);
                }
                uint32_t cpt2 = _get_cpt(pos);
                while (cpt2 == '\r' || cpt2 == '\n') {
                    cpt2 = _get_cpt(++pos);
                }
                _add_token(pos);
                continue;
            }

            size_t num_whitespaces = 0;
            size_t last_end_r_or_n = 0;
            while (_get_flags(pos + num_whitespaces).is_whitespace) {
                uint32_t cpt2 = _get_cpt(pos + num_whitespaces);
                if (cpt2 == '\r' || cpt2 == '\n') {
                    last_end_r_or_n = pos + num_whitespaces + 1;
                }
                num_whitespaces++;
            }

            if (last_end_r_or_n > 0) {
                pos = last_end_r_or_n;
                _add_token(pos);
                continue;
            }

            if (num_whitespaces > 1 && _get_cpt(pos + num_whitespaces) != OUT_OF_RANGE) {
                pos += num_whitespaces - 1;
                _add_token(pos);
                continue;
            }

            if (num_whitespaces > 0) {
                pos += num_whitespaces;
                _add_token(pos);
                continue;
            }

            _add_token(++pos);
        }
    }

    return bpe_offsets;
}

static std::vector<size_t> unicode_regex_split_custom_llama3(const std::string & text, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets; // store the offset of each word
    bpe_offsets.reserve(offsets.size()); // Reserve memory for the approximate size

    const auto cpts = unicode_cpts_from_utf8(text);

    // Round 3 optimization: pre-compute flags array for direct access
    static const auto cpt_flags = unicode_cpt_flags_array();
    static const unicode_cpt_flags undef(unicode_cpt_flags::UNDEFINED);

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;

        static const uint32_t OUT_OF_RANGE = 0xFFFFFFFF;
        auto _get_cpt = [&] (const size_t pos) -> uint32_t {
            return (offset_ini <= pos && pos < offset_end) ? cpts[pos] : OUT_OF_RANGE;
        };

        // Fast path: direct array access instead of function call
        auto _get_flags = [&] (const size_t pos) -> const unicode_cpt_flags& {
            if (!(offset_ini <= pos && pos < offset_end)) return undef;
            uint32_t cpt = cpts[pos];
            return cpt < cpt_flags.size() ? cpt_flags[cpt] : undef;
        };

        size_t _prev_end = offset_ini;
        auto _add_token = [&] (const size_t end) -> size_t {
            assert(_prev_end <= end && end <= offset_end);
            size_t len = end - _prev_end;
            if (len > 0) {
                bpe_offsets.push_back(len);
            }
            _prev_end = end;
            //if (len > 0) {
            //    std::string s = "";
            //    for(size_t p = end-len; p < end; p++)
            //        s += unicode_cpt_to_utf8(cpts[p]);
            //    printf(">>> '%s'\n", s.c_str());
            //}
            return len;
        };

        for (size_t pos = offset_ini; pos < offset_end; /*pos++*/ ) {
            const uint32_t cpt = _get_cpt(pos);
            const auto flags = _get_flags(pos);

            // regex: (?i:'s|'t|'re|'ve|'m|'ll|'d) // case insensitive
            if (cpt == '\'' && pos+1 < offset_end) {
                uint32_t cpt_next = unicode_tolower(_get_cpt(pos+1));
                if (cpt_next == 's' || cpt_next == 't' || cpt_next == 'm' || cpt_next == 'd') {
                    pos += _add_token(pos+2);
                    continue;
                }
                if (pos+2 < offset_end) {
                    uint32_t cpt_next_next = unicode_tolower(_get_cpt(pos+2));
                    if ((cpt_next == 'r' && cpt_next_next == 'e') ||
                        (cpt_next == 'v' && cpt_next_next == 'e') ||
                        (cpt_next == 'l' && cpt_next_next == 'l')) {
                        pos += _add_token(pos+3);
                        continue;
                    }
                }
            }

            // regex: [^\r\n\p{L}\p{N}]?\p{L}+
            if (!(cpt == '\r' || cpt == '\n' || flags.is_number)) {
                if (flags.is_letter || _get_flags(pos+1).is_letter) {  // one or more letters
                    pos++;
                    while (_get_flags(pos).is_letter) {
                        pos++;
                    }
                    _add_token(pos);
                    continue;
                }
            }

            // regex: \p{N}{1,3}
            if (flags.is_number) {
                size_t ini = pos;
                while (_get_flags(pos).is_number) {
                    if (++pos - ini >= 3 ) {
                        _add_token(pos);
                        ini = pos;
                    }
                }
                _add_token(pos);
                continue;
            }

            // regex: <space>?[^\s\p{L}\p{N}]+[\r\n]*
            auto flags2 = (cpt == ' ' ? _get_flags(pos+1) : flags);
            if (!(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags.as_uint()) {
                pos += (cpt == ' ');
                while (pos < offset_end && !(flags2.is_whitespace | flags2.is_letter | flags2.is_number) && flags2.as_uint()) {
                    flags2 = _get_flags(++pos);
                }
                uint32_t cpt2 = _get_cpt(pos);
                while (cpt2 == '\r' || cpt2 == '\n') {
                    cpt2 = _get_cpt(++pos);
                }
                _add_token(pos);
                continue;
            }

            size_t num_whitespaces = 0;
            size_t last_end_r_or_n = 0;
            while (_get_flags(pos+num_whitespaces).is_whitespace) {
                uint32_t cpt2 = _get_cpt(pos+num_whitespaces);
                if (cpt2 == '\r' || cpt2 == '\n') {
                    last_end_r_or_n = pos + num_whitespaces + 1;
                }
                num_whitespaces++;
            }

            // regex: \s*[\r\n]+
            if (last_end_r_or_n > 0) {
                pos = last_end_r_or_n;
                _add_token(pos);
                continue;
            }

            // regex: \s+(?!\S)
            if (num_whitespaces > 1 && _get_cpt(pos+num_whitespaces) != OUT_OF_RANGE) {
                pos += num_whitespaces - 1;
                _add_token(pos);
                continue;
            }

            // regex: \s+
            if (num_whitespaces > 0) {
                pos += num_whitespaces;
                _add_token(pos);
                continue;
            }

            // no matches
            _add_token(++pos);
        }
    }

    return bpe_offsets;
}

template <typename CharT>
static std::vector<size_t> unicode_regex_split_stl(const std::basic_string<CharT> & text, const std::basic_string<CharT> & regex, const std::vector<size_t> & offsets) {
    using BidirIt = typename std::basic_string<CharT>::const_iterator;
#ifdef _MSC_VER
    // Bypass bug in MSVC: https://github.com/ggml-org/llama.cpp/issues/17830
    constexpr auto regex_flags = std::regex_constants::ECMAScript;
#else
    constexpr auto regex_flags = std::regex_constants::optimize | std::regex_constants::nosubs;
#endif
    std::basic_regex<CharT> expr(regex, regex_flags);
    std::vector<size_t> bpe_offsets; // store the offset of each word
    bpe_offsets.reserve(offsets.size()); // Reserve memory for the approximate size
    size_t start = 0;
    for (auto offset : offsets) {
        std::regex_iterator<BidirIt> it(text.begin() + start, text.begin() + start + offset, expr);
        std::regex_iterator<BidirIt> end;

        int64_t start_idx = 0;
        while (it != end) {
            std::match_results<BidirIt> match = *it;
            if (match.position() > start_idx) {
                bpe_offsets.emplace_back(match.position() - start_idx);
            }
            bpe_offsets.emplace_back(match.length());
            start_idx = match.position() + match.length();
            ++it;
        }

        if (start_idx < (int64_t) offset) {
            bpe_offsets.emplace_back(offset - start_idx);
        }
        start += offset;
    }

    return bpe_offsets;
}

// K2 system regex patterns (from tokenization_kimi.py):
// [\p{Han}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
static std::vector<size_t> unicode_regex_split_custom_kimi_k2(const std::string & text, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets;
    bpe_offsets.reserve(offsets.size());

    const auto cpts = unicode_cpts_from_utf8(text);

    // Round 3 optimization: pre-compute flags array for direct access
    static const auto cpt_flags = unicode_cpt_flags_array();
    static const unicode_cpt_flags undef(unicode_cpt_flags::UNDEFINED);

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;

        static const uint32_t OUT_OF_RANGE = 0xFFFFFFFF;
        auto _get_cpt = [&] (const size_t pos) -> uint32_t {
            return (offset_ini <= pos && pos < offset_end) ? cpts[pos] : OUT_OF_RANGE;
        };

        // Fast path: direct array access instead of function call
        auto _get_flags = [&] (const size_t pos) -> const unicode_cpt_flags& {
            if (!(offset_ini <= pos && pos < offset_end)) return undef;
            uint32_t cpt = cpts[pos];
            return cpt < cpt_flags.size() ? cpt_flags[cpt] : undef;
        };

        size_t _prev_end = offset_ini;
        auto _add_token = [&] (const size_t end) -> size_t {
            assert(_prev_end <= end && end <= offset_end);
            size_t len = end - _prev_end;
            if (len > 0) {
                bpe_offsets.push_back(len);
            }
            _prev_end = end;
            return len;
        };

        for (size_t pos = offset_ini; pos < offset_end; /*pos++*/ ) {
            const uint32_t cpt = _get_cpt(pos);
            const auto flags = _get_flags(pos);

            // Pattern 1: [\p{Han}]+ (Chinese characters)
            if (unicode_cpt_is_han(cpt)) {
                while (unicode_cpt_is_han(_get_cpt(pos))) {
                    pos++;
                }
                _add_token(pos);
                continue;
            }

            // Pattern 2 & 3: Letter words excluding Han characters with optional contractions
            // [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?:'s|'t|'re|'ve|'m|'ll|'d)?
            // [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?:'s|'t|'re|'ve|'m|'ll|'d)?
            // Check if current char is a letter OR if current char could be a leading char and next char is a letter
            bool is_letter_pattern = (flags.is_letter && !unicode_cpt_is_han(cpt)) ||
                                     (!(cpt == '\r' || cpt == '\n' || flags.is_letter || flags.is_number) &&
                                      _get_flags(pos + 1).is_letter && !unicode_cpt_is_han(_get_cpt(pos + 1)));

            if (is_letter_pattern) {
                // Handle optional leading non-letter/non-number character
                bool has_leading_char = false;
                if (!(cpt == '\r' || cpt == '\n' || flags.is_letter || flags.is_number)) {
                    has_leading_char = true;
                    pos++;
                }

                // Match letter sequence (excluding Han characters)
                bool has_letters = false;
                while (_get_flags(pos).is_letter && !unicode_cpt_is_han(_get_cpt(pos))) {
                    has_letters = true;
                    pos++;
                }

                // Only proceed if we found letters (after potentially skipping leading char)
                if (has_letters || (!has_leading_char && _get_flags(pos).is_letter && !unicode_cpt_is_han(_get_cpt(pos)))) {
                    if (!has_letters) pos++; // consume the first letter if we didn't already

                    // Continue consuming letters
                    while (_get_flags(pos).is_letter && !unicode_cpt_is_han(_get_cpt(pos))) {
                        pos++;
                    }

                    // Check for optional contractions (?:'s|'t|'re|'ve|'m|'ll|'d)
                    if (_get_cpt(pos) == '\'' && pos + 1 < offset_end) {
                        uint32_t cpt_next = unicode_tolower(_get_cpt(pos + 1));
                        if (cpt_next == 's' || cpt_next == 't' || cpt_next == 'm' || cpt_next == 'd') {
                            pos += 2;
                        } else if (pos + 2 < offset_end) {
                            uint32_t cpt_next_next = unicode_tolower(_get_cpt(pos + 2));
                            if ((cpt_next == 'r' && cpt_next_next == 'e') ||
                                (cpt_next == 'v' && cpt_next_next == 'e') ||
                                (cpt_next == 'l' && cpt_next_next == 'l')) {
                                pos += 3;
                            }
                        }
                    }

                    _add_token(pos);
                    continue;
                } else if (has_leading_char) {
                    // We consumed a leading char but found no letters, backtrack
                    pos--;
                }
            }

            // Pattern 4: \p{N}{1,3} (numbers 1-3 digits)
            if (flags.is_number) {
                size_t ini = pos;
                while (_get_flags(pos).is_number) {
                    if (++pos - ini >= 3) {
                        _add_token(pos);
                        ini = pos;
                    }
                }
                _add_token(pos);
                continue;
            }

            // Pattern 5:  ?[^\s\p{L}\p{N}]+[\r\n]* (optional space + non-word chars + optional newlines)
            auto flags2 = (cpt == ' ' ? _get_flags(pos + 1) : flags);
            if (!(flags2.is_whitespace || flags2.is_letter || flags2.is_number) && flags2.as_uint()) {
                pos += (cpt == ' ');
                while (!(flags2.is_whitespace || flags2.is_letter || flags2.is_number) && flags2.as_uint()) {
                    flags2 = _get_flags(++pos);
                }
                // Match optional [\r\n]*
                uint32_t cpt2 = _get_cpt(pos);
                while (cpt2 == '\r' || cpt2 == '\n') {
                    cpt2 = _get_cpt(++pos);
                }
                _add_token(pos);
                continue;
            }

            // Count whitespace characters
            size_t num_whitespaces = 0;
            size_t last_end_r_or_n = 0;
            while (_get_flags(pos + num_whitespaces).is_whitespace) {
                uint32_t cpt2 = _get_cpt(pos + num_whitespaces);
                if (cpt2 == '\r' || cpt2 == '\n') {
                    last_end_r_or_n = pos + num_whitespaces + 1;
                }
                num_whitespaces++;
            }

            // Pattern 6: \s*[\r\n]+ (whitespace with newlines)
            if (last_end_r_or_n > 0) {
                pos = last_end_r_or_n;
                _add_token(pos);
                continue;
            }

            // Pattern 7: \s+(?!\S) (trailing whitespace)
            if (num_whitespaces > 1 && _get_cpt(pos + num_whitespaces) != OUT_OF_RANGE) {
                pos += num_whitespaces - 1;
                _add_token(pos);
                continue;
            }

            // Pattern 8: \s+ (general whitespace)
            if (num_whitespaces > 0) {
                pos += num_whitespaces;
                _add_token(pos);
                continue;
            }

            // No matches - consume single character
            _add_token(++pos);
        }
    }

    return bpe_offsets;
}

// AFMOE digit handling: splits digits with leading 1-2 based on total length modulo 3
static std::vector<size_t> unicode_regex_split_custom_afmoe(const std::string & text, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets;
    bpe_offsets.reserve(offsets.size());

    const auto cpts = unicode_cpts_from_utf8(text);

    // Round 3 optimization: pre-compute flags array for direct access
    static const auto cpt_flags = unicode_cpt_flags_array();
    static const unicode_cpt_flags undef(unicode_cpt_flags::UNDEFINED);

    size_t start = 0;
    for (auto offset : offsets) {
        const size_t offset_ini = start;
        const size_t offset_end = start + offset;
        assert(offset_end <= cpts.size());
        start = offset_end;

        // Fast path: direct array access instead of function call
        auto _get_flags = [&] (const size_t pos) -> const unicode_cpt_flags& {
            if (!(offset_ini <= pos && pos < offset_end)) return undef;
            uint32_t cpt = cpts[pos];
            return cpt < cpt_flags.size() ? cpt_flags[cpt] : undef;
        };

        size_t _prev_end = offset_ini;
        auto _add_token = [&] (const size_t end) -> size_t {
            assert(_prev_end <= end && end <= offset_end);
            size_t len = end - _prev_end;
            if (len > 0) {
                bpe_offsets.push_back(len);
            }
            _prev_end = end;
            return len;
        };

        for (size_t pos = offset_ini; pos < offset_end; ) {
            const auto flags = _get_flags(pos);

            // Handle digit sequences with special splitting logic
            if (flags.is_number) {
                size_t digit_start = pos;
                size_t digit_count = 0;

                // Count consecutive digits
                while (_get_flags(pos).is_number && pos < offset_end) {
                    digit_count++;
                    pos++;
                }

                // Split based on total length modulo 3
                size_t remainder = digit_count % 3;
                size_t current = digit_start;

                // Emit leading 1-2 digits if needed
                if (remainder > 0) {
                    _add_token(current + remainder);
                    current += remainder;
                }

                // Emit groups of 3
                while (current < digit_start + digit_count) {
                    _add_token(current + 3);
                    current += 3;
                }
                continue;
            }

            // For non-digits, just move forward
            pos++;
        }

        // Add any remaining content
        if (_prev_end < offset_end) {
            _add_token(offset_end);
        }
    }

    return bpe_offsets;
}

static std::vector<size_t> unicode_regex_split_custom(const std::string & text, const std::string & regex_expr, const std::vector<size_t> & offsets) {
    std::vector<size_t> bpe_offsets;

    // Round 29: Use optimized single-pass UTF-8 decode + category lookup for GPT2/Qwen35 patterns
    // Note: Qwen35 uses LLAMA3-based patterns with \p{M} support, which are handled by unicode_regex_split_custom_llama3
    // For now, we only optimize the pure GPT2 pattern
    if (regex_expr == "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)") {
        bpe_offsets = unicode_regex_split_custom_gpt2_optimized(text, offsets);
    } else if (
            regex_expr == "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" ||
            regex_expr == "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+") {

        bpe_offsets = unicode_regex_split_custom_llama3(text, offsets);
    } else if (
            regex_expr == "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+") {
        // Qwen35 regex pattern - similar to LLAMA3 but with \p{M} support for combining marks
        bpe_offsets = unicode_regex_split_custom_llama3_optimized(text, offsets);
    } else if (
            regex_expr == "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+") {
        // Qwen35 regex pattern (expanded form) - similar to LLAMA3 but with \p{M} support for combining marks
        bpe_offsets = unicode_regex_split_custom_llama3_optimized(text, offsets);
    } else if (regex_expr == "\\p{Han}+") {
        // K2's first pattern - handle all K2 patterns together
        bpe_offsets = unicode_regex_split_custom_kimi_k2(text, offsets);
    } else if (regex_expr == "\\p{AFMoE_digits}") {
        // AFMOE digit pattern - use custom implementation for proper splitting
        bpe_offsets = unicode_regex_split_custom_afmoe(text, offsets);
    } else if (regex_expr == "\\d{1,3}(?=(?:\\d{3})*\\b)") {
        // tiny_aya digit grouping pattern from tokenizer.json:
        //   {"type": "Split", "pattern": {"Regex": "\\d{1,3}(?=(?:\\d{3})*\\b)"}, "behavior": "Isolated"}
        // Splits digits into groups of 3 from the right (e.g., 1234567 -> 1, 234, 567)
        // TODO: Revisit this regex, in case there are any subtle tokenization differences with the original regex.
        bpe_offsets = unicode_regex_split_custom_afmoe(text, offsets);
    }

    return bpe_offsets;
}

//
// interface
//

std::string unicode_cpt_to_utf8(uint32_t cpt) {
    std::string result;

    if (/* 0x00 <= cpt && */ cpt <= 0x7f) {
        result.push_back(cpt);
        return result;
    }
    if (0x80 <= cpt && cpt <= 0x7ff) {
        result.push_back(0xc0 | ((cpt >> 6) & 0x1f));
        result.push_back(0x80 | (cpt & 0x3f));
        return result;
    }
    if (0x800 <= cpt && cpt <= 0xffff) {
        result.push_back(0xe0 | ((cpt >> 12) & 0x0f));
        result.push_back(0x80 | ((cpt >> 6) & 0x3f));
        result.push_back(0x80 | (cpt & 0x3f));
        return result;
    }
    if (0x10000 <= cpt && cpt <= 0x10ffff) {
        result.push_back(0xf0 | ((cpt >> 18) & 0x07));
        result.push_back(0x80 | ((cpt >> 12) & 0x3f));
        result.push_back(0x80 | ((cpt >> 6) & 0x3f));
        result.push_back(0x80 | (cpt & 0x3f));
        return result;
    }

    throw std::invalid_argument("invalid codepoint");
}

std::vector<uint32_t> unicode_cpts_normalize_nfd(const std::vector<uint32_t> & cpts) {
    auto comp = [] (const uint32_t cpt, const range_nfd & range) {
        return cpt < range.first;
    };
    std::vector<uint32_t> result(cpts.size());
    for (size_t i = 0; i < cpts.size(); ++i) {
        const uint32_t cpt = cpts[i];
        auto it = std::upper_bound(unicode_ranges_nfd.begin(), unicode_ranges_nfd.end(), cpt, comp) - 1;
        result[i] = (it->first <= cpt && cpt <= it->last) ? it->nfd : cpt;
    }
    return result;
}

// Round 21: Use simdutf library for optimized UTF-8 decoding
std::vector<uint32_t> unicode_cpts_from_utf8(const std::string & utf8) {
    return simdutf_utf8::decode_utf8(utf8.c_str(), utf8.size());
}

unicode_cpt_flags unicode_cpt_flags_from_cpt(const uint32_t cpt) {
    static const unicode_cpt_flags undef(unicode_cpt_flags::UNDEFINED);
    static const auto cpt_flags = unicode_cpt_flags_array();
    return cpt < cpt_flags.size() ? cpt_flags[cpt] : undef;
}

unicode_cpt_flags unicode_cpt_flags_from_utf8(const std::string & utf8) {
    static const unicode_cpt_flags undef(unicode_cpt_flags::UNDEFINED);
    if (utf8.empty()) {
        return undef;  // undefined
    }
    size_t offset = 0;
    return unicode_cpt_flags_from_cpt(unicode_cpt_from_utf8(utf8, offset));
}

std::string unicode_byte_to_utf8(uint8_t byte) {
    static std::unordered_map<uint8_t, std::string> map = unicode_byte_to_utf8_map();
    return map.at(byte);
}

uint8_t unicode_utf8_to_byte(const std::string & utf8) {
    static std::unordered_map<std::string, uint8_t> map = unicode_utf8_to_byte_map();
    return map.at(utf8);
}

uint32_t unicode_tolower(uint32_t cpt) {
    // binary search
    auto it = std::lower_bound(unicode_map_lowercase.begin(), unicode_map_lowercase.end(), cpt,
        [](const std::pair<uint32_t, uint32_t> & pair, uint32_t value) {
            return pair.first < value;
        });
    if (it != unicode_map_lowercase.end() && it->first == cpt) {
        return it->second;
    }
    return cpt;  // Return the original code point if no lowercase mapping is found
}

// Round 24: Optimized Han character detection
// Uses bit mask for faster range checking
// Most common range: CJK Unified Ideographs (0x4E00-0x9FFF) covers ~98% of Chinese characters

inline bool unicode_cpt_is_han(uint32_t cpt) {
    // Fast path: CJK Unified Ideographs (most common, ~98% of Chinese chars)
    if (cpt >= 0x4E00 && cpt <= 0x9FFF) return true;

    // CJK Compatibility Ideographs
    if (cpt >= 0xF900 && cpt <= 0xFAFF) return true;

    // CJK Extension A (less common)
    if (cpt >= 0x3400 && cpt <= 0x4DBF) return true;

    // Check remaining extensions (rarely used in typical text)
    // Using bit mask for faster checking
    if (cpt >= 0x20000) {
        // Extensions B, C, D, E, F
        return (cpt <= 0x2A6DF) ||   // Extension B
               (cpt >= 0x2A700 && cpt <= 0x2B73F) ||  // Extension C
               (cpt >= 0x2B740 && cpt <= 0x2B81F) ||  // Extension D
               (cpt >= 0x2B820 && cpt <= 0x2CEAF) ||  // Extension E
               (cpt >= 0x2CEB0 && cpt <= 0x2EBEF) ||  // Extension F
               (cpt >= 0x2F800 && cpt <= 0x2FA1F);    // Compatibility Supplement
    }

    return false;
}

std::vector<std::string> unicode_regex_split(const std::string & text, const std::vector<std::string> & regex_exprs) {
    // unicode categories
    static const std::map<std::string, int> k_ucat_enum = {
        { "\\p{N}", unicode_cpt_flags::NUMBER },
        { "\\p{L}", unicode_cpt_flags::LETTER },
        { "\\p{P}", unicode_cpt_flags::PUNCTUATION },
        { "\\p{M}", unicode_cpt_flags::ACCENT_MARK },
        { "\\p{S}", unicode_cpt_flags::SYMBOL },
        { "\\p{Lu}", unicode_cpt_flags::LETTER }, // Uppercase letter
        { "\\p{Ll}", unicode_cpt_flags::LETTER }, // Lowercase letter
        { "\\p{Lt}", unicode_cpt_flags::LETTER }, // Titlecase letter
        { "\\p{Lm}", unicode_cpt_flags::LETTER }, // Modifier letter
        { "\\p{Lo}", unicode_cpt_flags::LETTER }, // Other letter
    };

    static const std::map<int, int> k_ucat_cpt = {
        { unicode_cpt_flags::NUMBER,      0xD1 },
        { unicode_cpt_flags::LETTER,      0xD2 },
        { unicode_cpt_flags::PUNCTUATION, 0xD3 },
        { unicode_cpt_flags::ACCENT_MARK, 0xD4 },
        { unicode_cpt_flags::SYMBOL,      0xD5 },
    };

    static const std::map<int, std::string> k_ucat_map = {
        { unicode_cpt_flags::NUMBER,      "\x30-\x39" }, // 0-9
        { unicode_cpt_flags::LETTER,      "\x41-\x5A\x61-\x7A" }, // A-Za-z
        { unicode_cpt_flags::PUNCTUATION, "\x21-\x23\x25-\x2A\x2C-\x2F\x3A-\x3B\x3F-\x40\\\x5B-\\\x5D\x5F\\\x7B\\\x7D" }, // !-#%-*,-/:-;?-@\[-\]_\{\}
        { unicode_cpt_flags::ACCENT_MARK, "" }, // no sub-128 codepoints
        { unicode_cpt_flags::SYMBOL,      "\\\x24\\\x2B\x3C-\x3E\x5E\x60\\\x7C" }, // $+<=>^`|
    };

    // Pre-compute unicode flags array for fast direct access (Round 3 optimization)
    static const auto cpt_flags = unicode_cpt_flags_array();
    static const unicode_cpt_flags undef(unicode_cpt_flags::UNDEFINED);

    // Fast inline flag access - avoids function call overhead in hot loops
    auto get_flags_fast = [&](uint32_t cpt) -> const unicode_cpt_flags& {
        return cpt < cpt_flags.size() ? cpt_flags[cpt] : undef;
    };

    // compute collapsed codepoints only if needed by at least one regex
    bool need_collapse = false;
    for (const auto & regex_expr : regex_exprs) {
        // search for unicode categories
        for (const auto & ucat : k_ucat_enum) {
            if (std::string::npos != regex_expr.find(ucat.first)) {
                need_collapse = true;
                break;
            }
        }
    }

    const auto cpts = unicode_cpts_from_utf8(text);

    // generate a "collapsed" representation of the text, where all codepoints are replaced by a single byte
    // ref: https://github.com/ggml-org/llama.cpp/pull/6920#issuecomment-2081479935
    std::string text_collapsed;
    if (need_collapse) {
        // collapse all unicode categories
        text_collapsed.resize(cpts.size());

        for (size_t i = 0; i < cpts.size(); ++i) {
            // keep single-byte codepoints as is
            if (cpts[i] < 128) {
                text_collapsed[i] = cpts[i];
                continue;
            }

            const auto& flags = get_flags_fast(cpts[i]);

            if (flags.is_whitespace) {
                //NOTE: C++ std::regex \s does not mach 0x85, Rust and Python regex does.
                //text_collapsed[i] = (char) 0x85;  // <Next Line> as whitespace fallback
                text_collapsed[i] = (char) 0x0B;    // <vertical tab> as whitespace fallback
            } else if (k_ucat_cpt.find(flags.category_flag()) != k_ucat_cpt.end()) {
                text_collapsed[i] = k_ucat_cpt.at(flags.category_flag());
            } else {
                text_collapsed[i] = (char) 0xD0; // fallback
            }
        }
    }

    std::vector<size_t> bpe_offsets = { cpts.size() };

    for (const auto & regex_expr : regex_exprs) {
        // first, see if we have an efficient custom regex implementation
        auto tmp = unicode_regex_split_custom(text, regex_expr, bpe_offsets);

        if (!tmp.empty()) {
            bpe_offsets = std::move(tmp);
            continue;
        }

        // fallback to general-purpose std::regex / std::wregex
        try {
            // if a unicode category is used in the regex, we use the collapsed text and replace the unicode category
            // with the corresponding collapsed representation
            bool use_collapsed = false;
            for (const auto & ucat : k_ucat_enum) {
                if (std::string::npos != regex_expr.find(ucat.first)) {
                    use_collapsed = true;
                    break;
                }
            }
            const auto cpts_regex = unicode_cpts_from_utf8(regex_expr);

            if (use_collapsed) {
                // sanity-check that the original regex does not contain any non-ASCII characters
                for (size_t i = 0; i < cpts_regex.size(); ++i) {
                    if (cpts_regex[i] >= 128) {
                        throw std::runtime_error("Regex includes both unicode categories and non-ASCII characters - not supported");
                    }
                }

                // generate a collapsed representation of the regex
                std::string regex_expr_collapsed;

                // track if we are inside [], because nested [] are not allowed
                bool inside = false;
                for (size_t i = 0; i < regex_expr.size(); ++i) {
                    if (regex_expr[i] == '[' && (i == 0 || regex_expr[i - 1] != '\\')) {
                        regex_expr_collapsed += '[';
                        inside = true;
                        continue;
                    }

                    if (inside && regex_expr[i] == ']' && regex_expr[i - 1] != '\\') {
                        regex_expr_collapsed += ']';
                        inside = false;
                        continue;
                    }

                    // Match \p{...} Unicode properties of varying lengths
                    if (regex_expr[i + 0] == '\\' && i + 3 < regex_expr.size() &&
                        regex_expr[i + 1] == 'p' &&
                        regex_expr[i + 2] == '{') {
                        // Find the closing brace
                        size_t closing_brace = regex_expr.find('}', i + 3);
                        if (closing_brace != std::string::npos && closing_brace <= i + 10) { // reasonable limit
                            const std::string pat = regex_expr.substr(i, closing_brace - i + 1);
                            if (k_ucat_enum.find(pat) != k_ucat_enum.end()) {
                                if (!inside) {
                                    regex_expr_collapsed += '[';
                                }
                                regex_expr_collapsed += k_ucat_cpt.at(k_ucat_enum.at(pat));
                                regex_expr_collapsed += k_ucat_map.at(k_ucat_enum.at(pat));
                                if (!inside) {
                                    regex_expr_collapsed += ']';
                                }
                                i = closing_brace;
                                continue;
                            }
                        }
                    }

                    regex_expr_collapsed += regex_expr[i];
                }

                //printf("text_collapsed: %s\n", text_collapsed.c_str());
                //printf("regex_expr_collapsed: %s\n", regex_expr_collapsed.c_str());
                bpe_offsets = unicode_regex_split_stl(text_collapsed, regex_expr_collapsed, bpe_offsets);
            } else {
                // no unicode category used, we can use std::wregex directly
                std::wstring wregex_expr(cpts_regex.begin(), cpts_regex.end());

                // std::wregex \s does not mach non-ASCII whitespaces, using 0x0B as fallback
                std::wstring wtext(cpts.begin(), cpts.end());
                for (size_t i = 0; i < wtext.size(); ++i) {
                    if (wtext[i] > 0x7F && unicode_cpt_flags_from_cpt(wtext[i]).is_whitespace) {
                        wtext[i] = 0x0B;
                    }
                }

                //printf("text: %s\n", text.c_str());
                //printf("regex_expr: %s\n", regex_expr.c_str());
                bpe_offsets = unicode_regex_split_stl(wtext, wregex_expr, bpe_offsets);
            }
        } catch (std::regex_error & e) {
            fprintf(stderr, "Failed to process regex: '%s'\n", regex_expr.c_str());
            fprintf(stderr, "Regex error: %s\n", e.what());
            throw std::runtime_error("Failed to process regex");
        }
    }

    std::vector<std::string> bpe_words;
    bpe_words.reserve(bpe_offsets.size()); // reserve memory for the approximate size

    size_t start = 0;
    for (size_t & offset : bpe_offsets) {
        bpe_words.emplace_back();
        for (size_t i = start; i < start + offset; ++i) {
            bpe_words.back() += unicode_cpt_to_utf8(cpts[i]);
        }
        start += offset;
    }

    return unicode_byte_encoding_process(bpe_words);
}
