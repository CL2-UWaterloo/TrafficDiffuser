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

def denormalize(hl_scenarios, mean, std, map_min, map_max, speed_scale=15):
    """Convert a normalized ``(x, y, speed)`` endpoint to world units."""
    denormed_scenario = hl_scenarios * std + mean
    position = (denormed_scenario[..., :2] + 1) * (map_max - map_min) / 2 + map_min
    speed = (denormed_scenario[..., 2] + 1) * speed_scale
    return position, speed
