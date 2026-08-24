# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from argparse import ArgumentTypeError


def parse_bool(value):
    """Parse modern boolean values and the project's legacy plot values."""
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {'true', '1', 'yes', 'plot'}:
        return True
    if normalized in {'false', '0', 'no', 'no_plot'}:
        return False
    raise ArgumentTypeError(f'Expected true or false, got {value!r}')
