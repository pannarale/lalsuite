"""
This module provides class definitions corresponding to the elements that
can be found in a LIGO Light Weight XML file.  It also provides a class
representing an entire LIGO Light Weight XML document, and a ContentHandler
class for use with SAX2 parsers.
"""

__author__ = "Kipp Cannon <kipp@gravity.phys.uwm.edu>"
__date__ = "$Date$"
__version__ = "$Revision$"


import re
import sys
from xml import sax


#
# =============================================================================
#
#                         Document Header, and Indent
#
# =============================================================================
#

Header = """<?xml version='1.0' encoding='utf-8' ?>
<!DOCTYPE LIGO_LW SYSTEM "http://ldas-sw.ligo.caltech.edu/doc/ligolwAPI/html/ligolw_dtd.txt">"""

Indent = "\t"


#
# =============================================================================
#
#                                Element Class
#
# =============================================================================
#

class ElementError(Exception):
	"""
	Base class for exceptions generated by elements.
	"""
	pass


class Element(object):
	"""
	Base class for all element types.
	"""
	tagName = None
	empty = False
	validattributes = []
	validchildren = []

	def __init__(self, attrs = sax.xmlreader.AttributesImpl({})):
		"""
		Construct an element.  The argument is a
		sax.xmlreader.AttributesImpl object (see the xml.sax
		documentation, but it's basically a dictionary-like thing)
		used to set the element attributes.
		"""
		for key in attrs.keys():
			if key not in self.validattributes:
				raise ElementError, "%s does not have attribute %s" % (self.tagName, key)
		self.parentNode = None
		self.attributes = attrs
		self.childNodes = []
		self.pcdata = None

	def compare(self, other):
		# can't override __cmp__() because that screws up the
		# removeChild() method.
		"""
		Two elements compare as equal if they generate equivalent
		markup.
		"""
		result = cmp(self.tagName, other.tagName)
		if not result:
			result = cmp(dict(self.attributes), dict(other.attributes))
			if not result:
				result = cmp(self.pcdata, other.pcdata)
				if not result:
					a = self.childNodes[:]
					a.sort()
					b = self.childNodes[:]
					b.sort()
					result = cmp(a, b)
		return result

	def start_tag(self):
		"""
		Generate the string for the element's start tag.
		"""
		s = "<" + self.tagName
		for keyvalue in self.attributes.items():
			s += " %s=\"%s\"" % keyvalue
		if self.empty:
			s += "/>"
		else:
			s += ">"
		return s

	def end_tag(self):
		"""
		Generate the string for the element's end tag.  This
		function returns None for empty elements (single-tag
		elements).
		"""
		if self.empty:
			return None
		else:
			return "</" + self.tagName + ">"

	def appendChild(self, child):
		"""
		Add a child to this element.  The child's parentNode
		attribute is updated, too.
		"""
		if self.empty:
			raise ElementError, "%s cannot have children" % self.tagName
		self.childNodes.append(child)
		child.parentNode = self

	def removeChild(self, child):
		"""
		Remove a child from this element.  The child element is
		returned, and it's parentNode element is reset.
		"""
		self.childNodes.remove(child)
		child.parentNode = None
		return child

	def getElements(self, filter):
		"""
		Return a list of elements below elem for which filter(element)
		returns True.
		"""
		l = reduce(lambda l, e: l + e.getElements(filter), self.childNodes, [])
		if filter(self):
			l += [self]
		return l

	def getElementsByTagName(self, tagName):
		return self.getElements(lambda e: e.tagName == tagName)

	def getChildrenByAttributes(self, attrs):
		l = []
		for c in self.childNodes:
			try:
				if reduce(lambda t, (k, v): t and (c.getAttribute(k) == v), attrs.iteritems(), True):
					l.append(c)
			except KeyError:
				pass
		return l

	def getAttribute(self, attrname):
		return self.attributes[attrname]

	def setAttribute(self, attrname, value):
		self.attributes[attrname] = str(value)

	def appendData(self, content):
		"""
		Add characters to the element's pcdata.
		"""
		if self.pcdata:
			self.pcdata += content
		else:
			self.pcdata = content

	def write(self, file = sys.stdout, indent = ""):
		"""
		Recursively write an element and it's children to a file.
		"""
		print >>file, indent + self.start_tag()
		if not self.empty:
			for c in self.childNodes:
				if c.tagName not in self.validchildren:
					raise ElementError, "invalid child %s for %s" % (c.tagName, self.tagName)
				c.write(file, indent + Indent)
			if self.pcdata:
				print >>file, self.pcdata
			print >>file, indent + self.end_tag()


#
# =============================================================================
#
#                        LIGO Light Weight XML Elements
#
# =============================================================================
#

class LIGO_LW(Element):
	"""
	LIGO_LW element.
	"""
	tagName = "LIGO_LW"
	validchildren = [u"LIGO_LW", u"Comment", u"Param", u"Table", u"Array", u"Stream", u"IGWDFrame", u"AdcData", u"AdcInterval", u"Time", u"Detector"]


class Comment(Element):
	"""
	Comment element.
	"""
	tagName = "Comment"

	def write(self, file = sys.stdout, indent = ""):
		if self.pcdata:
			print >>file, indent + self.start_tag() + self.pcdata + self.end_tag()
		else:
			print >>file, indent + self.start_tag() + self.end_tag()


class Param(Element):
	"""
	Param element.
	"""
	tagName = "Param"
	validchildren = [u"Comment"]
	validattributes = [u"Name", u"Type", u"Start", u"Scale", u"Unit", u"DataUnit"]


class Table(Element):
	"""
	Table element.
	"""
	tagName = "Table"
	validchildren = [u"Comment", u"Column", u"Stream"]
	validattributes = [u"Name", u"Type"]

	def appendChild(self, child):
		Element.appendChild(self, child)
		# check number and order of children
		ncomment = 0
		ncolumn = 0
		nstream = 0
		for child in self.childNodes:
			if child.tagName == "Comment":
				if ncomment:
					raise ElementError, "only one Comment allowed in Table"
				if ncolumn or nstream:
					raise ElementError, "Comment must come before Column(s) and Stream in Table"
				ncomment += 1
			elif child.tagName == "Column":
				if nstream:
					raise ElementError, "Column(s) must come before Stream in Table"
				ncolumn += 1
			else:
				if nstream:
					raise ElementError, "only one Stream allowed in Table"
				nstream += 1


class Column(Element):
	"""
	Column element.
	"""
	tagName = "Column"
	empty = True
	validattributes = [u"Name", u"Type", u"Unit"]


class Array(Element):
	"""
	Array element.
	"""
	tagName = "Array"
	validchildren = [u"Dim", u"Stream"]
	validattributes = [u"Name", u"Type", u"Unit"]

	def appendChild(self, child):
		Element.appendChild(self, child)
		# check number and order of children
		nstream = 0
		for child in self.childNodes:
			if child.tagName == "Dim":
				if nstream:
					raise ElementError, "Dim(s) must come before Stream in Array"
			else:
				if nstream:
					raise ElementError, "only one Stream allowed in Array"
				nstream += 1


class Dim(Element):
	"""
	Dim element.
	"""
	tagName = "Dim"
	validattributes = [u"Name", u"Unit", u"Start", u"Scale"]


class Stream(Element):
	"""
	Stream element.
	"""
	tagName = "Stream"
	validattributes = [u"Name", u"Type", u"Delimiter", u"Encoding", u"Content"]

	def __init__(self, attrs = sax.xmlreader.AttributesImpl({})):
		if not attrs.has_key("Type"):
			attrs._attrs["Type"] = "Local"
		if not attrs.has_key("Delimiter"):
			attrs._attrs["Delimiter"] = ","
		if attrs["Type"] not in ["Remote", "Local"]:
			raise ElementError, "invalid value %s for Stream attribute Type" % attrs["Type"]
		Element.__init__(self, attrs)


class IGWDFrame(Element):
	"""
	IGWDFrame element.
	"""
	tagName = "IGWDFrame"
	validchildren = [u"Comment", u"Param", u"Time", u"Detector", u"AdcData", u"LIGO_LW", u"Stream", u"Array", u"IGWDFrame"]
	validattributes = [u"Name"]


class Detector(Element):
	"""
	Detector element.
	"""
	tagName = "Detector"
	validchildren = [u"Comment", u"Param", u"LIGO_LW"]
	validattributes = [u"Name"]


class AdcData(Element):
	"""
	AdcData element.
	"""
	tagName = "AdcData"
	validchildren = [u"AdcData", u"Comment", u"Param", u"Time", u"LIGO_LW", u"Array"]
	validattributes = [u"Name"]


class AdcInterval(Element):
	"""
	AdcInterval element.
	"""
	tagName = "AdcInterval"
	validchildren = [u"AdcData", u"Comment", u"Time"]
	validattributes = [u"Name", u"StartTime", u"DeltaT"]


class Time(Element):
	"""
	Time element.
	"""
	tagName = "Time"
	validattributes = [u"Name", u"Type"]

	def __init__(self, attrs = sax.xmlreader.AttributesImpl({})):
		if not attrs.has_key("Type"):
			attrs._attrs["Type"] = "ISO-8601"
		if attrs["Type"] not in ["GPS", "Unix", "ISO-8601"]:
			raise ElementError, "invalid value %s for Time attribute Type" % attrs["Type"]
		Element.__init__(self, attrs)


class Document(Element):
	"""
	Description of a LIGO LW file.
	"""
	tagName = "Document"
	validchildren = [u"LIGO_LW"]

	def write(self, file = sys.stdout):
		"""
		Write the document.
		"""
		print >>file, Header
		for c in self.childNodes:
			if c.tagName not in self.validchildren:
				raise ElementError, "invalid child %s for %s" % (c.tagName, self.tagName)
			c.write(file)


#
# =============================================================================
#
#                             SAX Content Handler
#
# =============================================================================
#

class LIGOLWContentHandler(sax.handler.ContentHandler):
	"""
	ContentHandler class for parsing LIGO Light Weight documents with a
	SAX2-compliant parser.

	Example:
		from xml import sax
		import ligolw

		doc = ligolw.Document()
		handler = ligolw.LIGOLWContentHandler(doc)
		sax.parse(file("H2-POWER_S5-816526720-34.xml"), handler)
		doc.write()
	"""
	def __init__(self, document):
		"""
		Initialize the handler by pointing it to the Document object
		into which the parsed file will be loaded.
		"""
		self.document = document
		self.current = self.document

	def startAdcData(self, attrs):
		return AdcData(attrs)

	def endAdcData(self):
		pass

	def startAdcInterval(self, attrs):
		return AdcInterval(attrs)

	def endAdcInterval(self):
		pass

	def startArray(self, attrs):
		return Array(attrs)

	def endArray(self):
		pass

	def startColumn(self, attrs):
		return Column(attrs)

	def endColumn(self):
		pass

	def startComment(self, attrs):
		return Comment(attrs)

	def endComment(self):
		pass

	def startDetector(self, attrs):
		return Detector(attrs)

	def endDetector(self):
		pass

	def startDim(self, attrs):
		return Dim(attrs)

	def endDim(self):
		pass

	def startIGWDFrame(self, attrs):
		return IGWDFrame(attrs)

	def endIGWDFrame(self):
		pass

	def startLIGO_LW(self, attrs):
		return LIGO_LW(attrs)

	def endLIGO_LW(self):
		pass

	def startParam(self, attrs):
		return Param(attrs)

	def endParams(self):
		pass

	def startStream(self, attrs):
		return Stream(attrs)

	def endStream(self):
		pass

	def startTable(self, attrs):
		return Table(attrs)

	def endTable(self):
		pass

	def startTime(self, attrs):
		return Time(attrs)

	def endTime(self):
		pass

	def startElement(self, name, attrs):
		if name == AdcData.tagName:
			child = self.startAdcData(attrs)
		elif name == AdcInterval.tagName:
			child = self.startAdcInterval(attrs)
		elif name == Array.tagName:
			child = self.startArray(attrs)
		elif name == Column.tagName:
			child = self.startColumn(attrs)
		elif name == Comment.tagName:
			child = self.startComment(attrs)
		elif name == Detector.tagName:
			child = self.startDetector(attrs)
		elif name == Dim.tagName:
			child = self.startDim(attrs)
		elif name == IGWDFrame.tagName:
			child = self.startIGWDFrame(attrs)
		elif name == LIGO_LW.tagName:
			child = self.startLIGO_LW(attrs)
		elif name == Param.tagName:
			child = self.startParam(attrs)
		elif name == Stream.tagName:
			child = self.startStream(attrs)
		elif name == Table.tagName:
			child = self.startTable(attrs)
		elif name == Time.tagName:
			child = self.startTime(attrs)
		else:
			raise ElementError, "unknown element tag %s" % name
		self.current.appendChild(child)
		self.current = child

	def endElement(self, name):
		if name == AdcData.tagName:
			self.endAdcData()
		elif name == AdcInterval.tagName:
			self.endAdcInterval()
		elif name == Array.tagName:
			self.endArray()
		elif name == Column.tagName:
			self.endColumn()
		elif name == Comment.tagName:
			self.endComment()
		elif name == Detector.tagName:
			self.endDetector()
		elif name == Dim.tagName:
			self.endDim()
		elif name == IGWDFrame.tagName:
			self.endIGWDFrame()
		elif name == LIGO_LW.tagName:
			self.endLIGO_LW()
		elif name == Param.tagName:
			self.endParam()
		elif name == Stream.tagName:
			self.endStream()
		elif name == Table.tagName:
			self.endTable()
		elif name == Time.tagName:
			self.endTime()
		else:
			raise ElementError, "unknown element tag %s" % name
		self.current = self.current.parentNode

	def characters(self, content):
		"""
		Discard character data for all elements but Comments and
		Streams.
		"""
		if self.current.tagName in ["Comment", "Stream"]:
			self.current.appendData(content)


#
# =============================================================================
#
#                            Convenience Functions
#
# =============================================================================
#

def make_parser(handler, validate = False):
	"""
	Convenience function to construct a document parser with or without
	validation.  Note:  when validation is enabled, the LIGO LW DTD
	identified in the document header must be accessible --- document
	parsing will fail if an internet connection to the LDAS document
	server is not available.
	"""
	try:
		parser = sax.make_parser("xml.sax.drivers2.drv_pyexpat")
	except TypeError:
		parser = sax.make_parser()
	parser.setContentHandler(handler)
	parser.setFeature(sax.handler.feature_validation, validate)
	return parser
