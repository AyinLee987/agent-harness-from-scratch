"""A 100-tool kit for the tool-count-vs-accuracy scaling experiment.

Context: the standard advice on "how many tools before an LLM's tool-calling
accuracy degrades" is that noticeable drop-off starts around 15-20 tools and
gets severe past 50-100, especially once tools have overlapping names/
descriptions (see the Anthropic "Writing effective tools for agents" blog
post and public MCP tool-overload write-ups). This module builds a
deliberately *realistic* tool catalog to let ``tool_scaling_test.py``
measure that curve directly against a real LLM instead of citing numbers.

Design: ten categories of ten tools each. Within a category, tools are
intentionally close in name/purpose (e.g. ``celsius_to_fahrenheit`` vs
``fahrenheit_to_celsius``) -- this is what actually confuses tool selection
in production, more than raw count alone. The first five categories were the
original 50-tool kit; the second five extend the same categories' *spirit*
(more stats, more text formatting, more date arithmetic, more unit
conversions, more encoding) so the later half of the registry is just as
plausibly-real as the first, not padding.

    math_*       -- 10 arithmetic/statistics ops
    text_*       -- 10 string ops
    date_*       -- 10 date/time ops
    convert_*    -- 10 unit conversions
    data_*       -- 10 encoding/misc utility ops
    stat_*       -- 10 more statistics ops (extends math_*)
    format_*     -- 10 more text-formatting ops (extends text_*)
    calendar_*   -- 10 more date/calendar ops (extends date_*)
    measure_*    -- 10 more unit conversions (extends convert_*)
    encode_*     -- 10 more encoding/hashing ops (extends data_*)

Every tool is pure and dependency-free so the kit runs anywhere.
"""

from __future__ import annotations

import base64
import calendar as _calendar
import codecs
import hashlib
import json
import math
import random
import re
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
# Category 6: stat_* (10) -- extends math_*
# ---------------------------------------------------------------------------


@tool
def stat_median(numbers: str) -> str:
    """Compute the median of a comma-separated list of numbers.

    Args:
        numbers: Comma-separated numbers, e.g. '3,1,2'.
    """
    values = _parse_numbers(numbers)
    return _num(statistics.median(values)) if values else "No numbers given."


@tool
def stat_stdev(numbers: str) -> str:
    """Compute the sample standard deviation of a comma-separated list of numbers.

    Args:
        numbers: Comma-separated numbers, at least two values.
    """
    values = _parse_numbers(numbers)
    if len(values) < 2:
        return "Need at least two numbers."
    return _num(statistics.stdev(values))


@tool
def stat_variance(numbers: str) -> str:
    """Compute the sample variance of a comma-separated list of numbers.

    Args:
        numbers: Comma-separated numbers, at least two values.
    """
    values = _parse_numbers(numbers)
    if len(values) < 2:
        return "Need at least two numbers."
    return _num(statistics.variance(values))


@tool
def stat_sum(numbers: str) -> str:
    """Compute the sum of a comma-separated list of numbers.

    Args:
        numbers: Comma-separated numbers, e.g. '1,2,3,4'.
    """
    values = _parse_numbers(numbers)
    return _num(sum(values)) if values else "No numbers given."


@tool
def stat_product(numbers: str) -> str:
    """Compute the product of a comma-separated list of numbers.

    Args:
        numbers: Comma-separated numbers, e.g. '1,2,3,4'.
    """
    values = _parse_numbers(numbers)
    return _num(math.prod(values)) if values else "No numbers given."


@tool
def stat_range(numbers: str) -> str:
    """Compute the range (max minus min) of a comma-separated list of numbers.

    Args:
        numbers: Comma-separated numbers, e.g. '5,2,9,1'.
    """
    values = _parse_numbers(numbers)
    return _num(max(values) - min(values)) if values else "No numbers given."


@tool
def stat_count(numbers: str) -> str:
    """Count how many numbers are in a comma-separated list.

    Args:
        numbers: Comma-separated numbers, e.g. '5,2,9,1,3'.
    """
    return str(len(_parse_numbers(numbers)))


@tool
def stat_mode(numbers: str) -> str:
    """Find the most frequently occurring value in a comma-separated list of numbers.

    Args:
        numbers: Comma-separated numbers, e.g. '1,2,2,3'.
    """
    values = _parse_numbers(numbers)
    return _num(statistics.mode(values)) if values else "No numbers given."


@tool
def stat_percentile(numbers: str, percentile: float) -> str:
    """Compute a percentile (0-100) of a comma-separated list of numbers using linear interpolation.

    Args:
        numbers: Comma-separated numbers.
        percentile: Percentile to compute, between 0 and 100.
    """
    values = sorted(_parse_numbers(numbers))
    if not values:
        return "No numbers given."
    if not 0 <= percentile <= 100:
        return "percentile must be between 0 and 100."
    idx = percentile / 100 * (len(values) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    frac = idx - lo
    return _num(values[lo] + (values[hi] - values[lo]) * frac)


@tool
def stat_normalize(numbers: str) -> str:
    """Min-max normalize a comma-separated list of numbers to the 0-1 range.

    Args:
        numbers: Comma-separated numbers, e.g. '1,2,3,4,5'.
    """
    values = _parse_numbers(numbers)
    if not values:
        return "No numbers given."
    lo, hi = min(values), max(values)
    if lo == hi:
        return ",".join(_num(0) for _ in values)
    return ",".join(_num((v - lo) / (hi - lo)) for v in values)


# ---------------------------------------------------------------------------
# Category 7: format_* (10) -- extends text_*
# ---------------------------------------------------------------------------


@tool
def format_snake_case(text: str) -> str:
    """Convert text to snake_case.

    Args:
        text: The text to convert.
    """
    return "_".join(text.split()).lower()


@tool
def format_camel_case(text: str) -> str:
    """Convert text to camelCase.

    Args:
        text: The text to convert.
    """
    words = text.split()
    if not words:
        return ""
    return words[0].lower() + "".join(w.capitalize() for w in words[1:])


@tool
def format_kebab_case(text: str) -> str:
    """Convert text to kebab-case.

    Args:
        text: The text to convert.
    """
    return "-".join(text.split()).lower()


@tool
def format_capitalize_first(text: str) -> str:
    """Capitalize only the first letter of text, leaving the rest unchanged.

    Args:
        text: The text to convert.
    """
    return text[:1].upper() + text[1:] if text else text


@tool
def format_remove_punctuation(text: str) -> str:
    """Remove punctuation characters from text, keeping letters, digits, and spaces.

    Args:
        text: The text to strip.
    """
    return re.sub(r"[^\w\s]", "", text)


@tool
def format_truncate(text: str, max_length: int) -> str:
    """Truncate text to a maximum length, appending '...' if it was cut.

    Args:
        text: The text to truncate.
        max_length: Maximum number of characters to keep before the ellipsis.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


@tool
def format_pad_left(text: str, width: int, pad_char: str = " ") -> str:
    """Pad text on the left with a character until it reaches a given width.

    Args:
        text: The text to pad.
        width: Target total length.
        pad_char: Single character to pad with; default a space.
    """
    return text.rjust(width, pad_char or " ")


@tool
def format_pad_right(text: str, width: int, pad_char: str = " ") -> str:
    """Pad text on the right with a character until it reaches a given width.

    Args:
        text: The text to pad.
        width: Target total length.
        pad_char: Single character to pad with; default a space.
    """
    return text.ljust(width, pad_char or " ")


@tool
def format_repeat(text: str, times: int) -> str:
    """Repeat text a given number of times, concatenated with no separator.

    Args:
        text: The text to repeat.
        times: Number of repetitions.
    """
    return text * times


@tool
def format_slugify(text: str) -> str:
    """Convert text into a lowercase, hyphen-separated URL slug.

    Args:
        text: The text to slugify.
    """
    cleaned = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[\s]+", "-", cleaned)


# ---------------------------------------------------------------------------
# Category 8: calendar_* (10) -- extends date_*
# ---------------------------------------------------------------------------


@tool
def calendar_month_name(month: int) -> str:
    """Return the full name of a month given its number (1-12).

    Args:
        month: Month number, 1-12.
    """
    if not 1 <= month <= 12:
        return "month must be between 1 and 12."
    return datetime(2000, month, 1).strftime("%B")


@tool
def calendar_days_in_month(year: int, month: int) -> str:
    """Return how many days are in a given month of a given year.

    Args:
        year: The four-digit year.
        month: Month number, 1-12.
    """
    if not 1 <= month <= 12:
        return "month must be between 1 and 12."
    return str(_calendar.monthrange(year, month)[1])


@tool
def calendar_is_weekend(date: str) -> str:
    """Check whether a date falls on a Saturday or Sunday.

    Args:
        date: Date in YYYY-MM-DD format.
    """
    try:
        parsed = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return f"Could not parse date: {date}"
    return "true" if parsed.weekday() >= 5 else "false"


@tool
def calendar_next_weekday(date: str, weekday_name: str) -> str:
    """Find the next occurrence of a named weekday strictly after a date.

    Args:
        date: Date in YYYY-MM-DD format.
        weekday_name: Full weekday name, e.g. 'Friday'.
    """
    try:
        parsed = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return f"Could not parse date: {date}"
    names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    try:
        target = names.index(weekday_name.strip().capitalize())
    except ValueError:
        return f"Unknown weekday name: {weekday_name}"
    days_ahead = (target - parsed.weekday()) % 7
    days_ahead = days_ahead or 7
    return (parsed + timedelta(days=days_ahead)).strftime("%Y-%m-%d")


@tool
def calendar_week_number(date: str) -> str:
    """Return the ISO-8601 week number (1-53) for a date.

    Args:
        date: Date in YYYY-MM-DD format.
    """
    try:
        parsed = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return f"Could not parse date: {date}"
    return str(parsed.isocalendar()[1])


@tool
def calendar_start_of_month(date: str) -> str:
    """Return the first day of the month containing a date.

    Args:
        date: Date in YYYY-MM-DD format.
    """
    try:
        parsed = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return f"Could not parse date: {date}"
    return parsed.replace(day=1).strftime("%Y-%m-%d")


@tool
def calendar_end_of_month(date: str) -> str:
    """Return the last day of the month containing a date.

    Args:
        date: Date in YYYY-MM-DD format.
    """
    try:
        parsed = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return f"Could not parse date: {date}"
    last_day = _calendar.monthrange(parsed.year, parsed.month)[1]
    return parsed.replace(day=last_day).strftime("%Y-%m-%d")


@tool
def calendar_age_in_years(birth_date: str, as_of_date: str) -> str:
    """Compute a whole-years age given a birth date and an as-of date.

    Args:
        birth_date: Birth date in YYYY-MM-DD format.
        as_of_date: Date to compute the age as of, in YYYY-MM-DD format.
    """
    try:
        birth = datetime.strptime(birth_date, "%Y-%m-%d")
        as_of = datetime.strptime(as_of_date, "%Y-%m-%d")
    except ValueError:
        return "Could not parse one of the dates."
    years = as_of.year - birth.year
    if (as_of.month, as_of.day) < (birth.month, birth.day):
        years -= 1
    return str(years)


@tool
def calendar_subtract_days(date: str, days: int) -> str:
    """Subtract a number of days from a date.

    Args:
        date: Date in YYYY-MM-DD format.
        days: Number of days to subtract (must be non-negative).
    """
    try:
        parsed = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return f"Could not parse date: {date}"
    return (parsed - timedelta(days=days)).strftime("%Y-%m-%d")


@tool
def calendar_is_same_day(date_a: str, date_b: str) -> str:
    """Check whether two YYYY-MM-DD dates are the same calendar day.

    Args:
        date_a: First date in YYYY-MM-DD format.
        date_b: Second date in YYYY-MM-DD format.
    """
    return "true" if date_a.strip() == date_b.strip() else "false"


# ---------------------------------------------------------------------------
# Category 9: measure_* (10) -- extends convert_*
# ---------------------------------------------------------------------------


@tool
def measure_miles_to_nautical_miles(miles: float) -> str:
    """Convert a distance from statute miles to nautical miles.

    Args:
        miles: Distance in statute miles.
    """
    return _num(miles * 0.868976)


@tool
def measure_nautical_miles_to_miles(nautical_miles: float) -> str:
    """Convert a distance from nautical miles to statute miles.

    Args:
        nautical_miles: Distance in nautical miles.
    """
    return _num(nautical_miles * 1.15078)


@tool
def measure_acres_to_hectares(acres: float) -> str:
    """Convert an area from acres to hectares.

    Args:
        acres: Area in acres.
    """
    return _num(acres * 0.404686)


@tool
def measure_hectares_to_acres(hectares: float) -> str:
    """Convert an area from hectares to acres.

    Args:
        hectares: Area in hectares.
    """
    return _num(hectares * 2.47105)


@tool
def measure_sqft_to_sqm(square_feet: float) -> str:
    """Convert an area from square feet to square meters.

    Args:
        square_feet: Area in square feet.
    """
    return _num(square_feet * 0.092903)


@tool
def measure_sqm_to_sqft(square_meters: float) -> str:
    """Convert an area from square meters to square feet.

    Args:
        square_meters: Area in square meters.
    """
    return _num(square_meters * 10.7639)


@tool
def measure_mph_to_kmh(mph: float) -> str:
    """Convert a speed from miles per hour to kilometers per hour.

    Args:
        mph: Speed in miles per hour.
    """
    return _num(mph * 1.60934)


@tool
def measure_kmh_to_mph(kmh: float) -> str:
    """Convert a speed from kilometers per hour to miles per hour.

    Args:
        kmh: Speed in kilometers per hour.
    """
    return _num(kmh * 0.621371)


@tool
def measure_bytes_to_megabytes(byte_count: float) -> str:
    """Convert a size from bytes to decimal megabytes (1 MB = 1,000,000 bytes).

    Args:
        byte_count: Size in bytes.
    """
    return _num(byte_count / 1_000_000)


@tool
def measure_megabytes_to_bytes(megabytes: float) -> str:
    """Convert a size from decimal megabytes to bytes (1 MB = 1,000,000 bytes).

    Args:
        megabytes: Size in megabytes.
    """
    return _num(megabytes * 1_000_000)


# ---------------------------------------------------------------------------
# Category 10: encode_* (10) -- extends data_*
# ---------------------------------------------------------------------------


@tool
def encode_hex_encode(text: str) -> str:
    """Encode text as a hexadecimal string.

    Args:
        text: The text to encode.
    """
    return text.encode("utf-8").hex()


@tool
def encode_hex_decode(hex_string: str) -> str:
    """Decode a hexadecimal string back to text.

    Args:
        hex_string: The hexadecimal string to decode.
    """
    try:
        return bytes.fromhex(hex_string).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        return f"Could not decode hex: {exc}"


@tool
def encode_rot13(text: str) -> str:
    """Apply the ROT13 substitution cipher to text.

    Args:
        text: The text to transform.
    """
    return codecs.encode(text, "rot_13")


@tool
def encode_caesar_cipher(text: str, shift: int) -> str:
    """Encrypt text with a Caesar cipher, shifting letters by a fixed amount.

    Args:
        text: The text to encrypt.
        shift: Number of positions to shift each letter (may be negative).
    """
    return _caesar_shift(text, shift)


@tool
def encode_caesar_decipher(text: str, shift: int) -> str:
    """Decrypt text that was encrypted with :func:`encode_caesar_cipher` using the same shift.

    Args:
        text: The text to decrypt.
        shift: The shift amount originally used to encrypt it.
    """
    return _caesar_shift(text, -shift)


@tool
def encode_md5_hash(text: str) -> str:
    """Compute the MD5 checksum of text, as a hex digest.

    Args:
        text: The text to hash.
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()


@tool
def encode_sha256_hash(text: str) -> str:
    """Compute the SHA-256 checksum of text, as a hex digest.

    Args:
        text: The text to hash.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@tool
def encode_count_bytes(text: str) -> str:
    """Count the number of UTF-8 encoded bytes in text (may exceed the character count).

    Args:
        text: The text to measure.
    """
    return str(len(text.encode("utf-8")))


@tool
def encode_is_ascii(text: str) -> str:
    """Check whether text contains only ASCII characters.

    Args:
        text: The text to check.
    """
    return "true" if text.isascii() else "false"


@tool
def encode_strip_html_tags(text: str) -> str:
    """Remove HTML tags from text, leaving only the visible content.

    Args:
        text: The HTML text to strip.
    """
    return re.sub(r"<[^>]+>", "", text)


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


def _caesar_shift(text: str, shift: int) -> str:
    out = []
    for ch in text:
        if ch.isalpha():
            base = ord("A") if ch.isupper() else ord("a")
            out.append(chr((ord(ch) - base + shift) % 26 + base))
        else:
            out.append(ch)
    return "".join(out)


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
    # stat_* (10)
    stat_median, stat_stdev, stat_variance, stat_sum, stat_product,
    stat_range, stat_count, stat_mode, stat_percentile, stat_normalize,
    # format_* (10)
    format_snake_case, format_camel_case, format_kebab_case,
    format_capitalize_first, format_remove_punctuation, format_truncate,
    format_pad_left, format_pad_right, format_repeat, format_slugify,
    # calendar_* (10)
    calendar_month_name, calendar_days_in_month, calendar_is_weekend,
    calendar_next_weekday, calendar_week_number, calendar_start_of_month,
    calendar_end_of_month, calendar_age_in_years, calendar_subtract_days,
    calendar_is_same_day,
    # measure_* (10)
    measure_miles_to_nautical_miles, measure_nautical_miles_to_miles,
    measure_acres_to_hectares, measure_hectares_to_acres,
    measure_sqft_to_sqm, measure_sqm_to_sqft, measure_mph_to_kmh,
    measure_kmh_to_mph, measure_bytes_to_megabytes, measure_megabytes_to_bytes,
    # encode_* (10)
    encode_hex_encode, encode_hex_decode, encode_rot13, encode_caesar_cipher,
    encode_caesar_decipher, encode_md5_hash, encode_sha256_hash,
    encode_count_bytes, encode_is_ascii, encode_strip_html_tags,
]

assert len(ALL_TOOLS) == 100, f"Expected 100 tools, got {len(ALL_TOOLS)}"
assert len({t.name for t in ALL_TOOLS}) == 100, "Duplicate tool names in ALL_TOOLS"


def build_registry(n: int = 100) -> ToolRegistry:
    """Build a registry with the first ``n`` tools from :data:`ALL_TOOLS`.

    Used by the scaling experiment to grow the tool set while keeping the
    subset deterministic and reproducible.
    """
    if not 1 <= n <= len(ALL_TOOLS):
        raise ValueError(f"n must be between 1 and {len(ALL_TOOLS)}.")
    return ToolRegistry(ALL_TOOLS[:n])
