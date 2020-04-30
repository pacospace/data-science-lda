#!/usr/bin/env python3
# thamos
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

"""Exception for Data Science LDA."""


class UnknownFileTypeError(Exception):
    """Exception error when file type not known is requested to be retrieved."""


class InputFileMissingError(Exception):
    """Exception error when a required input is not provided."""


class InputPathFileMissingError(Exception):
    """Exception error when in the path provided there is no file."""


class NotKnownDatasetMethodsError(Exception):
    """Exception error when dataset methods enums are not known."""


class LdaInputTypeError(Exception):
    """Exception error when an LDA input for a parameter is not correct."""
