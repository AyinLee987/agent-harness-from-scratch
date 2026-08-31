"""A 50-tool kit for the tool-count-vs-accuracy scaling experiment.

Context: the standard advice on "how many tools before an LLM's tool-calling
accuracy degrades" is that noticeable drop-off starts around 15-20 tools and
gets severe past 50-100, especially once tools have overlapping names/
descriptions (see the Anthropic "Writing effective tools for agents" blog
post and public MCP tool-overload write-ups). This module builds a
deliberately *realistic* 50-tool catalog to let ``tool_scaling_test.py``
measure that curve directly against a real LLM instead of citing numbers.

Design: five categories of ten tools each. Within a category, tools are
intentionally close in name/purpose (e.g. ``celsius_to_fahrenheit`` vs
``fahrenheit_to_celsius``) -- this is what actually confuses tool selection
in production, more than raw count alone.

    math_*       -- 10 arithmetic/statistics ops
    text_*       -- 10 string ops
    date_*       -- 10 date/time ops
    convert_*    -- 10 unit conversions
    data_*       -- 10 encoding/misc utility ops

Every tool is pure and dependency-free so the kit runs anywhere.
"""

from __future__ import annotations

import base64
import json
import math
import random
import statistics
import urllib.parse
import uuid
from datetime import datetime, timedelta, timezone
from typing import List

from agent import ToolRegistry, tool

# ---------------------------------------------------------------------------
# Category 1: math_* (10)
# ---------------------------------------------------------------------------


@tool
def math_add(a: float, b: float) -> str:
    """Add two numbers together.

    Args:
        a: First number.
        b: Second number.
    """
    return _num(a + b)


@tool
def math_subtract(a: float, b: float) -> str:
    """Subtract the second number from the first.

    Args:
        a: Number to subtract from.
        b: Number to subtract.
    """
    return _num(a - b)


@tool
def math_multiply(a: float, b: float) -> str:
    """Multiply two numbers together.

    Args:
        a: First number.
        b: Second number.
    """
    return _num(a * b)


@tool
def math_divide(a: float, b: float) -> str:
    """Divide the first number by the second.

    Args:
        a: Dividend.
        b: Divisor (must be nonzero).
    """
    if b == 0:
        return "Cannot divide by zero."
    return _num(a / b)


@tool
def math_power(base: float, exponent: float) -> str:
    """Raise a number to a power.

    Args:
        base: The base number.
        exponent: The exponent.
    """
    return _num(base**exponent)


@tool
def math_sqrt(value: float) -> str:
    """Compute the square root of a non-negative number.

    Args:
        value: The number to take the square root of.
    """
    if value < 0:
        return "Cannot take the square root of a negative number."
    return _num(math.sqrt(value))


@tool
def math_modulo(a: float, b: float) -> str:
    """Compute the remainder of dividing the first number by the second.

    Args:
        a: Dividend.
        b: Divisor (must be nonzero).
    """
    if b == 0:
        return "Cannot compute modulo with a zero divisor."
    return _num(a % b)


@tool
def math_average(numbers: str) -> str:
    """Compute the arithmetic mean of a comma-separated list of numbers.

    Args:
        numbers: Comma-separated numbers, e.g. '1,2,3,4'.
    """
    values = _parse_numbers(numbers)
    return _num(statistics.mean(values)) if values else "No numbers given."


@tool
def math_min_max(numbers: str) -> str:
    """Find the minimum and maximum of a comma-separated list of numbers.

    Args:
        numbers: Comma-separated numbers, e.g. '5,2,9,1'.
    """
    values = _parse_numbers(numbers)
    if not values:
        return "No numbers given."
    return f"min={_num(min(values))} max={_num(max(values))}"


@tool
def math_round(value: float, digits: int = 0) -> str:
    """Round a number to a given number of decimal digits.

    Args:
        value: The number to round.
        digits: Decimal digits to keep; default 0.
    """
    return _num(round(value, digits))


# ---------------------------------------------------------------------------
# Category 2: text_* (10)
# ---------------------------------------------------------------------------


@tool
def text_uppercase(text: str) -> str:
    """Convert text to uppercase.

    Args:
        text: The text to convert.
    """
    return text.upper()


@tool
def text_lowercase(text: str) -> str:
    """Convert text to lowercase.

    Args:
        text: The text to convert.
    """
    return text.lower()


@tool
def text_reverse(text: str) -> str:
    """Reverse the characters of a string.

    Args:
        text: The text to reverse.
    """
    return text[::-1]


@tool
def text_word_count(text: str) -> str:
    """Count the number of whitespace-separated words in text.

    Args:
        text: The text to count words in.
    """
    return str(len(text.split()))


@tool
def text_char_count(text: str) -> str:
    """Count the total number of characters in text, including spaces.

    Args:
        text: The text to count characters in.
    """
    return str(len(text))


@tool
def text_trim(text: str) -> str:
    """Strip leading and trailing whitespace from text.

    Args:
        text: The text to trim.
    """
    return text.strip()


@tool
def text_title_case(text: str) -> str:
    """Convert text to Title Case (capitalize each word).

    Args:
        text: The text to convert.
    """
    return text.title()


@tool
def text_replace(text: str, old: str, new: str) -> str:
    """Replace every occurrence of a substring with another substring.

    Args:
        text: The text to modify.
        old: Substring to find.
        new: Substring to substitute in.
    """
    return text.replace(old, new)


@tool
def text_is_palindrome(text: str) -> str:
    """Check whether text reads the same forwards and backwards (ignoring case/spaces).

    Args:
        text: The text to check.
    """
    cleaned = "".join(ch.lower() for ch in text if ch.isalnum())
    return "true" if cleaned == cleaned[::-1] else "false"


@tool
def text_count_vowels(text: str) -> str:
    """Count the number of vowels (a, e, i, o, u) in text.

    Args:
        text: The text to scan.
    """
    return str(sum(1 for ch in text.lower() if ch in "aeiou"))


# ---------------------------------------------------------------------------
# Category 3: date_* (10)
# ---------------------------------------------------------------------------


@tool
def date_current_date(_unused: str = "") -> str:
    """Return today's date in UTC as YYYY-MM-DD."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


@tool
def date_current_time(_unused: str = "") -> str:
    """Return the current time of day in UTC as HH:MM:SS."""
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


@tool
def date_day_of_week(date: str) -> str:
    """Return the day of the week for a date.

    Args:
        date: Date in YYYY-MM-DD format.
    """
    try:
        return datetime.strptime(date, "%Y-%m-%d").strftime("%A")
    except ValueError:
        return f"Could not parse date: {date}"


@tool
def date_add_days(date: str, days: int) -> str:
    """Add a number of days to a date.

    Args:
        date: Date in YYYY-MM-DD format.
        days: Number of days to add (may be negative).
    """
    try:
        parsed = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return f"Could not parse date: {date}"
    return (parsed + timedelta(days=days)).strftime("%Y-%m-%d")


@tool
def date_days_between(start_date: str, end_date: str) -> str:
    """Compute the number of days between two dates.

    Args:
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.
    """
    try:
        a = datetime.strptime(start_date, "%Y-%m-%d")
        b = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        return "Could not parse one of the dates."
    return str((b - a).days)


@tool
def date_is_leap_year(year: int) -> str:
    """Check whether a given year is a leap year.

    Args:
        year: The four-digit year.
    """
    is_leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    return "true" if is_leap else "false"


@tool
def date_timestamp_to_date(timestamp: int) -> str:
    """Convert a Unix timestamp (seconds) to a UTC date string.

    Args:
        timestamp: Unix timestamp in seconds.
    """
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


@tool
def date_to_timestamp(date: str) -> str:
    """Convert a YYYY-MM-DD date to a Unix timestamp (seconds, UTC midnight).

    Args:
        date: Date in YYYY-MM-DD format.
    """
    try:
        parsed = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return f"Could not parse date: {date}"
    return str(int(parsed.timestamp()))


@tool
def date_format(date: str, output_format: str) -> str:
    """Reformat a YYYY-MM-DD date using a strftime pattern.

    Args:
        date: Date in YYYY-MM-DD format.
        output_format: A strftime pattern, e.g. '%d/%m/%Y'.
    """
    try:
        parsed = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return f"Could not parse date: {date}"
    try:
        return parsed.strftime(output_format)
    except ValueError as exc:
        return f"Invalid format string: {exc}"


@tool
def date_quarter(date: str) -> str:
    """Return which calendar quarter (Q1-Q4) a date falls in.

    Args:
        date: Date in YYYY-MM-DD format.
    """
    try:
        parsed = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return f"Could not parse date: {date}"
    return f"Q{(parsed.month - 1) // 3 + 1}"


# ---------------------------------------------------------------------------
# Category 4: convert_* (10)
# ---------------------------------------------------------------------------


@tool
def convert_celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a temperature from Celsius to Fahrenheit.

    Args:
        celsius: Temperature in Celsius.
    """
    return _num(celsius * 9 / 5 + 32)


@tool
def convert_fahrenheit_to_celsius(fahrenheit: float) -> str:
    """Convert a temperature from Fahrenheit to Celsius.

    Args:
        fahrenheit: Temperature in Fahrenheit.
    """
    return _num((fahrenheit - 32) * 5 / 9)


@tool
def convert_km_to_miles(kilometers: float) -> str:
    """Convert a distance from kilometers to miles.

    Args:
        kilometers: Distance in kilometers.
    """
    return _num(kilometers * 0.621371)


@tool
def convert_miles_to_km(miles: float) -> str:
    """Convert a distance from miles to kilometers.

    Args:
        miles: Distance in miles.
    """
    return _num(miles * 1.60934)


@tool
def convert_kg_to_lbs(kilograms: float) -> str:
    """Convert a mass from kilograms to pounds.

    Args:
        kilograms: Mass in kilograms.
    """
    return _num(kilograms * 2.20462)


@tool
def convert_lbs_to_kg(pounds: float) -> str:
    """Convert a mass from pounds to kilograms.

    Args:
        pounds: Mass in pounds.
    """
    return _num(pounds * 0.453592)


@tool
def convert_meters_to_feet(meters: float) -> str:
    """Convert a length from meters to feet.

    Args:
        meters: Length in meters.
    """
    return _num(meters * 3.28084)


@tool
def convert_feet_to_meters(feet: float) -> str:
    """Convert a length from feet to meters.

    Args:
        feet: Length in feet.
    """
    return _num(feet * 0.3048)


@tool
def convert_liters_to_gallons(liters: float) -> str:
    """Convert a volume from liters to US gallons.

    Args:
        liters: Volume in liters.
    """
    return _num(liters * 0.264172)


@tool
def convert_gallons_to_liters(gallons: float) -> str:
    """Convert a volume from US gallons to liters.

    Args:
        gallons: Volume in US gallons.
    """
    return _num(gallons * 3.78541)


# ---------------------------------------------------------------------------
# Category 5: data_* (10)
# ---------------------------------------------------------------------------


@tool
def data_base64_encode(text: str) -> str:
    """Encode text as Base64.

    Args:
        text: The text to encode.
    """
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


@tool
def data_base64_decode(encoded: str) -> str:
    """Decode a Base64 string back to text.

    Args:
        encoded: The Base64 string to decode.
    """
    try:
        return base64.b64decode(encoded).decode("utf-8")
    except Exception as exc:  # noqa: BLE001 - surface to the model
        return f"Could not decode Base64: {exc}"


@tool
def data_url_encode(text: str) -> str:
    """URL-encode (percent-encode) text.

    Args:
        text: The text to encode.
    """
    return urllib.parse.quote(text)


@tool
def data_url_decode(encoded: str) -> str:
    """URL-decode a percent-encoded string.

    Args:
        encoded: The percent-encoded string to decode.
    """
    return urllib.parse.unquote(encoded)


@tool
def data_generate_uuid(_unused: str = "") -> str:
    """Generate a random UUID (version 4)."""
    return str(uuid.uuid4())


@tool
def data_random_number(minimum: int = 0, maximum: int = 100) -> str:
    """Generate a random integer within an inclusive range.

    Args:
        minimum: Lower bound, inclusive.
        maximum: Upper bound, inclusive.
    """
    if minimum > maximum:
        return "minimum must be <= maximum."
    return str(random.randint(minimum, maximum))


@tool
def data_json_validate(text: str) -> str:
    """Check whether text is valid JSON.

    Args:
        text: The text to validate.
    """
    try:
        json.loads(text)
        return "valid"
    except json.JSONDecodeError as exc:
        return f"invalid: {exc}"


@tool
def data_sort_list(items: str, descending: bool = False) -> str:
    """Sort a comma-separated list of items alphabetically.

    Args:
        items: Comma-separated items, e.g. 'banana,apple,cherry'.
        descending: Sort descending instead of ascending.
    """
    values = [item.strip() for item in items.split(",") if item.strip()]
    values.sort(reverse=descending)
    return ",".join(values)


@tool
def data_dedupe_list(items: str) -> str:
    """Remove duplicate entries from a comma-separated list, preserving order.

    Args:
        items: Comma-separated items, e.g. 'a,b,a,c,b'.
    """
    seen: set[str] = set()
    result: List[str] = []
    for item in (i.strip() for i in items.split(",")):
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return ",".join(result)


@tool
def data_char_frequency(text: str, character: str) -> str:
    """Count how many times a single character appears in text.

    Args:
        text: The text to scan.
        character: The single character to count.
    """
    return str(text.count(character))


# ---------------------------------------------------------------------------
# Helpers + registry builder
# ---------------------------------------------------------------------------


def _num(value: float) -> str:
    """Render a float without a trailing .0 for whole numbers."""
    return str(int(value)) if float(value).is_integer() else str(round(value, 6))


def _parse_numbers(raw: str) -> List[float]:
    values: List[float] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            values.append(float(chunk))
        except ValueError:
            continue
    return values


# Declared in a fixed order so build_registry(n) takes a stable, reproducible
# first-N slice -- this is what the scaling experiment varies.
ALL_TOOLS = [
    # math_* (10)
    math_add, math_subtract, math_multiply, math_divide, math_power,
    math_sqrt, math_modulo, math_average, math_min_max, math_round,
    # text_* (10)
    text_uppercase, text_lowercase, text_reverse, text_word_count,
    text_char_count, text_trim, text_title_case, text_replace,
    text_is_palindrome, text_count_vowels,
    # date_* (10)
    date_current_date, date_current_time, date_day_of_week, date_add_days,
    date_days_between, date_is_leap_year, date_timestamp_to_date,
    date_to_timestamp, date_format, date_quarter,
    # convert_* (10)
    convert_celsius_to_fahrenheit, convert_fahrenheit_to_celsius,
    convert_km_to_miles, convert_miles_to_km, convert_kg_to_lbs,
    convert_lbs_to_kg, convert_meters_to_feet, convert_feet_to_meters,
    convert_liters_to_gallons, convert_gallons_to_liters,
    # data_* (10)
    data_base64_encode, data_base64_decode, data_url_encode, data_url_decode,
    data_generate_uuid, data_random_number, data_json_validate,
    data_sort_list, data_dedupe_list, data_char_frequency,
]

assert len(ALL_TOOLS) == 50, f"Expected 50 tools, got {len(ALL_TOOLS)}"
assert len({t.name for t in ALL_TOOLS}) == 50, "Duplicate tool names in ALL_TOOLS"


def build_registry(n: int = 50) -> ToolRegistry:
    """Build a registry with the first ``n`` tools from :data:`ALL_TOOLS`.

    Used by the scaling experiment to grow the tool set while keeping the
    subset deterministic and reproducible.
    """
    if not 1 <= n <= len(ALL_TOOLS):
        raise ValueError(f"n must be between 1 and {len(ALL_TOOLS)}.")
    return ToolRegistry(ALL_TOOLS[:n])
