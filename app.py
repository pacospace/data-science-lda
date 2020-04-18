
#!/usr/bin/env python3
# 
# Copyright(C) 2020 Francesco Murdaca
#
# This program is free software: you can redistribute it and / or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

"""Main file for Data Science LDA."""

import logging

from create_srcopsmetrics_inputs import create_source_ops_metrics_inputs

_LOGGER = logging.getLogger(__name__)

if __name__ == "__main__":
    _LOGGER.info("Creating inputs for SrcOpsMetrics...")
    create_source_ops_metrics_inputs()

    _LOGGER.info("Collecting README files using SrcOpsMetrics...")