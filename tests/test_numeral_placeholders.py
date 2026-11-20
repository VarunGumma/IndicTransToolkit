"""Numbers must be masked as one placeholder and restored unchanged.

Before the fix, Indian-grouped amounts and rupee/paise amounts were masked only in part
("₹7,302.34" -> "₹ 7 , <ID1>", "₹1,37,11,477" -> "₹ <ID1> , 477"), so IndicTrans2 copied the
leftover digits next to the restored ones and translations came out as "₹7, 302.34" and
"₹1,37,11, 477". Short decimals were not masked at all and "8.5%" came back as "8. 5".
"""
import re

import pytest

from IndicTransToolkit.processor import IndicProcessor

SRC, TGT = "eng_Latn", "guj_Gujr"


@pytest.fixture()
def ip():
    return IndicProcessor(inference=True)


def pre(ip, text):
    out = ip.preprocess_batch([text], src_lang=SRC, tgt_lang=TGT)[0]
    return out[len(f"{SRC} {TGT} "):]


def digits_outside_placeholders(s):
    return re.findall(r"\d", re.sub(r"<\s*ID\d+\s*>", "", s))


def round_trip(ip, text):
    """Pre-process, then post-process the masked text as if the model had copied it."""
    masked = pre(ip, text)
    return masked, ip.postprocess_batch([masked], lang=SRC)[0]


@pytest.mark.parametrize(
    "text, number",
    [
        ("Your current balance is ₹7,302.34.", "7,302.34"),  # thousands + paise
        ("An amount of ₹1,234.50 has been debited.", "1,234.50"),
        ("Please transfer ₹1,37,11,477 to complete the booking.", "1,37,11,477"),  # crore
        ("A penalty of Rs. 84,32,45,364 will be charged.", "84,32,45,364"),
        ("Inflation rose to 10.75% last month.", "10.75%"),
    ],
)
def test_number_masked_whole_and_restored(ip, text, number):
    masked, restored = round_trip(ip, text)
    assert digits_outside_placeholders(masked) == [], masked
    assert number in restored, restored


def test_masking_a_number_leaves_longer_numbers_intact(ip):
    text = "Pay ₹2,34,567 now and ₹12,34,567 later."
    masked, restored = round_trip(ip, text)
    assert digits_outside_placeholders(masked) == [], masked
    assert "₹2,34,567" in restored and "₹12,34,567" in restored, restored


def test_unrelated_entities_unchanged(ip):
    masked, restored = round_trip(ip, "Visit www.example.com or mail a.b@test.org, info@x.in today.")
    assert "@" not in masked and "example" not in masked, masked
    for s in ("www.example.com", "a.b@test.org", "info@x.in"):
        assert s in restored, restored


def test_short_numbers_still_left_to_the_model(ip):
    assert pre(ip, "There are 45 beds.") == "There are 45 beds ."
    assert pre(ip, "GST of 8.5% will be applied.") == "GST of 8.5 % will be applied ."
