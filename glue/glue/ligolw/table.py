# $Id$
#
# Copyright (C) 2006  Kipp C. Cannon
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#


"""
While the ligolw module provides classes and parser support for reading and
writing LIGO Light Weight XML documents, this module supplements that code
with classes and parsers that add intelligence to the in-RAM document
representation.

In particular, the document tree associated with a Table element is
enhanced.  During parsing, the Stream element in this module converts the
character data contained within it into a list of objects.  The list
contains one object for each row of the table, and the objects' attributes
are the names of the table's columns.  When the document is written out
again, the Stream element serializes the row objects back into character
data.

The Table element exports a list-like interface to the rows.  The Column
elements also provide list-like access to the values in the corresponding
columns of the table.
"""

__author__ = "Kipp Cannon <kipp@gravity.phys.uwm.edu>"
__date__ = "$Date$"[7:-2]
__version__ = "$Revision$"[11:-2]

import copy
import re
import sys
from xml.sax.saxutils import escape as xmlescape
from xml.sax.xmlreader import AttributesImpl

import ligolw
import tokenizer
import types
import ilwd


#
# =============================================================================
#
#                           Column Name Manipulation
#
# =============================================================================
#


# Regular expression to extract the significant part of a column name
# according to the LIGO LW naming conventions.

# FIXME: the pattern should be
#
# r"(?:\A[a-z0-9_]+:|\A)(?P<FullName>(?:[a-z0-9_]+:|\A)(?P<Name>[a-z0-9_]+))\Z"
#
# but people are putting upper case letters in names!!!!!  Someone is going
# to get the beats.


ColumnPattern = re.compile(r"(?:\A\w+:|\A)(?P<FullName>(?:(?P<Table>\w+):|\A)(?P<Name>\w+))\Z")


def StripColumnName(name):
	"""
	Return the significant portion of a column name according to LIGO
	LW naming conventions.
	"""
	try:
		return ColumnPattern.search(name).group("Name")
	except AttributeError:
		return name


def CompareColumnNames(name1, name2):
	"""
	Convenience function to compare two column names according to LIGO
	LW naming conventions.
	"""
	return cmp(StripColumnName(name1), StripColumnName(name2))


def getColumnsByName(elem, name):
	"""
	Return a list of columns with name name under elem.
	"""
	return elem.getElements(lambda e: (e.tagName == ligolw.Column.tagName) and (CompareColumnNames(e.getAttribute("Name"), name) == 0))


#
# =============================================================================
#
#                           Table Name Manipulation
#
# =============================================================================
#


# Regular expression used to extract the signifcant portion of a table or
# stream name, according to LIGO LW naming conventions.


TablePattern = re.compile(r"(?:\A[a-z0-9_]+:|\A)(?P<Name>[a-z0-9_]+):table\Z")


def StripTableName(name):
	"""
	Return the significant portion of a table name according to LIGO LW
	naming conventions.
	"""
	try:
		return TablePattern.search(name).group("Name")
	except AttributeError:
		return name


def CompareTableNames(name1, name2):
	"""
	Convenience function to compare two table names according to LIGO
	LW naming conventions.
	"""
	return cmp(StripTableName(name1), StripTableName(name2))


def getTablesByName(elem, name):
	"""
	Return a list of tables with name name under elem.
	"""
	return elem.getElements(lambda e: (e.tagName == ligolw.Table.tagName) and (CompareTableNames(e.getAttribute("Name"), name) == 0))


#
# =============================================================================
#
#                                  Utilities
#
# =============================================================================
#


def new_from_template(template):
	"""
	Construct a new Table document subtree whose structure is the same
	as the template table, that is it has the same columns etc..  The
	rows are not copied.  Note that a fair amount of metadata is shared
	between the original and new tables.  In particular, a copy of the
	Table object itself is created (but with no rows), and copies of
	the child nodes are created.  All other object references are
	shared between the two instances, such as the RowType attribute on
	the Table object.
	"""
	new = copy.copy(template)
	new.childNodes = map(copy.copy, template.childNodes)
	for child in new.childNodes:
		child.parentNode = new
	del new[:]
	new._end_of_columns()
	return new


def get_table(xmldoc, name):
	"""
	Scan xmldoc for a table named name.  Raises ValueError if not
	exactly 1 such table is found.
	"""
	tables = getTablesByName(xmldoc, name)
	if len(tables) != 1:
		raise ValueError, "document must contain exactly one %s table" % StripTableName(name)
	return tables[0]


def reassign_ids(elem):
	"""
	Recurse over all tables below elem which possess ID generators, and
	use the generators to assign new IDs to the rows in each table,
	recording the modifications in a mapping of old row keys to new row
	keys.  Finally, apply the mapping to all rows of all tables.
	"""
	mapping = {}
	for tbl in elem.getElementsByTagName(ligolw.Table.tagName):
		if tbl.ids is not None:
			tbl.updateKeyMapping(mapping)
	for tbl in elem.getElementsByTagName(ligolw.Table.tagName):
		tbl.applyKeyMapping(mapping)


#
# =============================================================================
#
#                                Column Element
#
# =============================================================================
#


class Column(ligolw.Column):
	"""
	High-level column element that provides list-like access to the
	values in a column.
	"""
	def __init__(self, attrs):
		ligolw.Column.__init__(self, attrs)
		self.asattribute = StripColumnName(self.getAttribute("Name"))

	def __len__(self):
		"""
		Return the number of values in this column.
		"""
		return len(self.parentNode)

	def __getitem__(self, i):
		"""
		Retrieve the value in this column in row i.
		"""
		if type(i) == slice:
			return map(lambda r: getattr(r, self.asattribute), self.parentNode[i])
		else:
			return getattr(self.parentNode[i], self.asattribute)

	def __setitem__(self, i, value):
		"""
		Set the value in this column in row i.
		"""
		if type(i) == slice:
			map(lambda r: setattr(r, self.asattribute, value), self.parentNode[i])
		else:
			setattr(self.parentNode[i], self.asattribute, value)

	def __iter__(self):
		"""
		Return an iterator object for iterating over values in this
		column.
		"""
		for row in self.parentNode:
			yield getattr(row, self.asattribute)

	def count(self, value):
		"""
		Return the number of rows with this column equal to value.
		"""
		n = 0
		for row in self.parentNode:
			if getattr(row, self.asattribute) == value:
				n += 1
		return n

	def index(self, value):
		"""
		Return the smallest index of the row(s) with this column
		equal to value.
		"""
		for i in xrange(len(self.parentNode)):
			if getattr(self.parentNode[i], self.asattribute) == value:
				return i
		raise ValueError, value

	def __contains__(self, value):
		"""
		Returns True or False if there is or is not, respectively,
		a row containing val in this column.
		"""
		for i in xrange(len(self.parentNode)):
			if getattr(self.parentNode[i], self.asattribute) == value:
				return True
		return False

	def asarray(self):
		"""
		Construct a numpy array from this column.  Note that this
		creates a copy of the data, so modifications made to the
		array will not be recorded in the original document.
		"""
		if self.getAttribute("Type") in types.StringTypes:
			raise TypeError, "Column does not have numeric type"
		import numpy
		return numpy.fromiter(self, dtype = types.ToNumPyType[self.getAttribute("Type")])


#
# =============================================================================
#
#                                Stream Element
#
# =============================================================================
#


class InterningRowBuilder(tokenizer.RowBuilder):
	"""
	This subclass of the tokenizer.RowBuilder class respects the
	"interning" hints provided by table definitions, and attempts to
	replace the values of row attributes associated with interned
	columns with references to shared instances of those values.  This
	results in a reduction in memory use which is small for most
	documents, but can be subtantial when dealing with poorly-designed
	tables containing large volumes of repeated information.
	
	The values are stored in a dictionary that is shared between all
	instances of this class, and which survives forever.  Nothing is
	ever naturally "uninterned", so the string dictionary grows without
	bound as more documents are processed.  This can be a problem in
	some use cases, and the work-around is to run

	>>> InterningRowBuilder.strings.clear()

	to reset the dictionary and appropriate points in the application.
	"""
	strings = {}
	def append(self, tokens):
		for row in tokenizer.RowBuilder.append(self, tokens):
			for col in self.interns:
				val = getattr(row, col)
				setattr(row, col, self.strings.setdefault(val, val))
			yield row


#
# Select the RowBuilder class to use when parsing tables.
#


RowBuilder = tokenizer.RowBuilder


class TableStream(ligolw.Stream):
	"""
	High-level Stream element for use inside Tables.  This element
	knows how to parse the delimited character stream into rows in the
	parent element, and knows how to turn the parent's rows back into a
	character stream.
	"""
	def __init__(self, attrs):
		ligolw.Stream.__init__(self, attrs)
		self.__tokenizer = tokenizer.Tokenizer(self.getAttribute("Delimiter"))
		self.__rowbuilder = None

	def config(self, parentNode):
		# some initialization that requires access to the
		# parentNode, and so cannot be done inside the __init__()
		# function.
		self.__tokenizer.set_types([(parentNode.loadcolumns is None or colname in parentNode.loadcolumns or None) and pytype for pytype, colname in zip(parentNode.columnpytypes, parentNode.columnnames)])
		columnnames = [name for name in parentNode.columnnames if parentNode.loadcolumns is None or name in parentNode.loadcolumns]
		interncolumns = [name for name in (parentNode.interncolumns or tuple()) if name in columnnames]
		self.__rowbuilder = RowBuilder(parentNode.RowType, columnnames, interncolumns)
		return self

	def appendData(self, content):
		# tokenize buffer, pack into row objects, and append to
		# table
		for row in self.__rowbuilder.append(self.__tokenizer.append(content)):
			self.parentNode.append(row)

	def unlink(self):
		"""
		Break internal references within the document tree rooted
		on this element to promote garbage collection.
		"""
		self.__tokenizer = None
		self.__rowbuilder = None
		ligolw.Stream.unlink(self)

	def write(self, file = sys.stdout, indent = u""):
		rowfmt = unicode(indent + ligolw.Indent + self.getAttribute("Delimiter").join([types.ToFormat[c.getAttribute("Type")] for c in self.parentNode.getElementsByTagName(ligolw.Column.tagName)]))
		colnames = self.parentNode.columnnames

		# loop over parent's rows.  This is complicated because we
		# need to not put a delimiter at the end of the last row.
		file.write(self.start_tag(indent) + u"\n")
		rowiter = iter(self.parentNode)
		try:
			row = rowiter.next()
			# FIXME: in Python 2.5, use attrgetter(*colnames)
			# for attribute tuplizing
			file.write(xmlescape(rowfmt % tuple([getattr(row, name) for name in colnames])))
			rowfmt = unicode(self.getAttribute("Delimiter") + u"\n" + rowfmt)
			while True:
				row = rowiter.next()
				file.write(xmlescape(rowfmt % tuple([getattr(row, name) for name in colnames])))
		except StopIteration:
			if len(self.parentNode) > 0:
				file.write(u"\n")
		file.write(self.end_tag(indent) + u"\n")

	# FIXME: This function is for the metaio library:  metaio cares
	# what order the attributes of XML tags come in.  This function
	# will be removed when the metaio library is fixed.
	def start_tag(self, indent):
		"""
		Generate the element start tag.
		"""
		return indent + u"<%s Name=\"%s\" Type=\"%s\" Delimiter=\"%s\">" % (self.tagName, self.getAttribute("Name"), self.getAttribute("Type"), self.getAttribute("Delimiter"))


#
# =============================================================================
#
#                                Table Element
#
# =============================================================================
#


class TableRow(object):
	"""
	Helpful parent class for row objects.  Also used as the default row
	class by Table instances.
	"""
	pass


class Table(ligolw.Table, list):
	"""
	High-level Table element that knows about its columns and rows.
	"""
	validcolumns = None
	loadcolumns = None
	interncolumns = None
	constraints = None
	how_to_index = None
	RowType = TableRow
	ids = None

	def __init__(self, *attrs):
		"""
		Initialize
		"""
		ligolw.Table.__init__(self, *attrs)
		self.columnnames = []
		self.columntypes = []
		self.columnpytypes = []

	#
	# Sequence methods
	#

	def filterRows(self, func):
		"""
		Delete all rows for which func(row) evaluates to False.
		"""
		for i in xrange(len(self) - 1, -1, -1):
			if not func(self[i]):
				del self[i]
		return self


	#
	# Column access
	#

	def getColumnByName(self, name):
		"""
		Retrieve and return the Column child element whose name is
		as given.  Raises KeyError if this table has no column by
		that name.
		"""
		try:
			return getColumnsByName(self, name)[0]
		except IndexError:
			raise KeyError, name


	def appendColumn(self, name):
		"""
		Append a column named "name" to the table.  Returns the new
		child.  Raises ValueError if the table already has a column
		by that name, and KeyError if the validcolumns attribute of
		this table does not contain an entry for a column by that
		name.
		"""
		if getColumnsByName(self, name):
			raise ValueError, "duplicate Column \"%s\"" % name
		column = Column(AttributesImpl({u"Name": "%s:%s" % (StripTableName(self.tableName), name), u"Type": self.validcolumns[name]}))
		streams = self.getElementsByTagName(ligolw.Stream.tagName)
		if streams:
			self.insertBefore(column, streams[0])
		else:
			self.appendChild(column)
		return column


	#
	# Element methods
	#

	def _update_column_info(self):
		"""
		Used for validation during parsing, and additional
		book-keeping.  For internal use only.
		"""
		del self.columnnames[:]
		del self.columntypes[:]
		del self.columnpytypes[:]
		for child in self.childNodes:
			if child.tagName != ligolw.Column.tagName:
				continue
			colname = StripColumnName(child.getAttribute("Name"))
			llwtype = child.getAttribute("Type")
			if self.validcolumns is not None:
				if colname not in self.validcolumns.keys():
					raise ligolw.ElementError, "invalid Column '%s' for Table '%s'" % (child.getAttribute("Name"), self.getAttribute("Name"))
				if self.validcolumns[colname] != llwtype:
					raise ligolw.ElementError, "invalid type '%s' for Column '%s' in Table '%s'" % (llwtype, child.getAttribute("Name"), self.getAttribute("Name"))
			if colname in self.columnnames:
				raise ligolw.ElementError, "duplicate Column '%s'" % child.getAttribute("Name")
			self.columnnames.append(colname)
			self.columntypes.append(llwtype)
			try:
				self.columnpytypes.append(types.ToPyType[llwtype])
			except KeyError:
				raise ligolw.ElementError, "unrecognized Type '%s' for Column '%s' in Table '%s'" % (llwtype, child.getAttribute("Name"), self.getAttribute("Name"))

	def _verifyChildren(self, i):
		"""
		Used for validation during parsing, and additional
		book-keeping.  For internal use only.
		"""
		ligolw.Table._verifyChildren(self, i)
		child = self.childNodes[i]
		if child.tagName == ligolw.Column.tagName:
			self._update_column_info()
		elif child.tagName == ligolw.Stream.tagName:
			if child.getAttribute("Name") != self.getAttribute("Name"):
				raise ligolw.ElementError, "Stream name '%s' does not match Table name '%s'" % (child.getAttribute("Name"), self.getAttribute("Name"))

	def _end_of_columns(self):
		"""
		Called during parsing to indicate that the last Column
		child element has been added.
		"""
		pass

	def _end_of_rows(self):
		"""
		Called during parsing to indicate that the last row has
		been added.
		"""
		pass

	def removeChild(self, child):
		"""
		Remove a child from this element.  The child element is
		returned, and it's parentNode element is reset.
		"""
		ligolw.Table.removeChild(self, child)
		if child.tagName == ligolw.Column.tagName:
			self._update_column_info()
		return child

	def unlink(self):
		"""
		Break internal references within the document tree rooted
		on this element to promote garbage collection.
		"""
		ligolw.Table.unlink(self)
		del self[:]


	#
	# Row ID manipulation
	#

	def sync_ids(self):
		"""
		Determines the highest-numbered ID in this table, and sets
		the counter for the table's row ID generator so as to cause
		it to yield the next ID in the sequence and higher.  If the
		generator is already set to yield an ID greater than the
		max found, then it is left unmodified.  The return value is
		the row ID generator object.  If the table does not possess
		a row ID generator, then this function is a no-op.

		Note that tables of the same name typically share
		references to the same ID generator so that IDs can be
		generated that are unique across all tables.  To set the
		generator to produce an ID greater than that in any table,
		sync_ids() needs to be run on every table sharing the
		generator.
		"""
		if self.ids is not None:
			n = 0
			for id in self.getColumnByName(self.ids.column_name):
				n = max(n, ilwd.ILWDID(id) + 1)
			if n > self.ids.n:
				self.ids.n = n
		return self.ids

	def updateKeyMapping(self, mapping):
		"""
		Used as the first half of the row key reassignment
		algorithm.  Accepts a dictionary mapping old key --> new
		key.  Iterates over the rows in this table, using the
		table's own row ID generator to assign a new key to each
		row, recording the changes in the mapping.  Returns the
		mapping.  Raises ValueError if the table has no ID
		generator.
		"""
		if self.ids is None:
			raise ValueError, self
		try:
			column = self.getColumnByName(self.ids.column_name)
		except KeyError:
			# table is missing its ID column
			return mapping
		for i, old in enumerate(column):
			if old in mapping:
				column[i] = mapping[old]
			else:
				column[i] = mapping[old] = self.ids.next()
		return mapping

	def applyKeyMapping(self, mapping):
		"""
		Used as the second half of the key reassignment algorithm.
		Loops over each row in the table, replacing references to
		old row keys with the new values from the mapping.
		"""
		for coltype, colname in zip(self.columntypes, self.columnnames):
			if coltype in types.IDTypes and (self.ids is None or colname != self.ids.column_name):
				column = self.getColumnByName(colname)
				for i, old in enumerate(column):
					if old in mapping:
						column[i] = mapping[old]


#
# =============================================================================
#
#                               Content Handler
#
# =============================================================================
#


#
# Override portions of the ligolw.LIGOLWContentHandler class
#


__parent_startStream = ligolw.LIGOLWContentHandler.startStream
__parent_endStream = ligolw.LIGOLWContentHandler.endStream


def startColumn(self, attrs):
	return Column(attrs)


def startStream(self, attrs):
	if self.current.tagName == ligolw.Table.tagName:
		self.current._end_of_columns()
		return TableStream(attrs).config(self.current)
	return __parent_startStream(self, attrs)


def endStream(self):
	# stream tokenizer uses delimiter to identify end of each token, so
	# add a final delimiter to induce the last token to get parsed.
	# Also call _end_of_rows() hook.
	if self.current.parentNode.tagName == ligolw.Table.tagName:
		self.current.appendData(self.current.getAttribute("Delimiter"))
		self.current.parentNode._end_of_rows()
	else:
		__parent_endStream(self)


def startTable(self, attrs):
	return Table(attrs)


def endTable(self):
	# Table elements are allowed to contain 0 Stream children, but
	# _end_of_columns() and _end_of_rows() hooks must be called
	# regardless, so we do that here if needed.
	if self.current.childNodes[-1].tagName != ligolw.Stream.tagName:
		self.current._end_of_columns()
		self.current._end_of_rows()


ligolw.LIGOLWContentHandler.startColumn = startColumn
ligolw.LIGOLWContentHandler.startStream = startStream
ligolw.LIGOLWContentHandler.endStream = endStream
ligolw.LIGOLWContentHandler.startTable = startTable
ligolw.LIGOLWContentHandler.endTable = endTable
