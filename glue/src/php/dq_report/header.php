<table width=100%>	   <tr>	   	   <td width=20%><a href="http://www.ligo.caltech.edu">     	   	   <img src="../seginsert/img/LIGOLogo.gif" border="0"></a></td>		   <td><center><h2><font face="arial"><?php echo $uTitle ?></font></center></h2></td>		   <td width=20%><right><a href="http://www.ligo.org">		       <img align="right" src="../seginsert/img/lsc.gif" border="0"></right></td>	   </tr></table>

<p>
<font color="blue">
This page uses the "--report" function in ligolw_dq_query tool to print out which flags are defined/undefined at the given time(s). <br>
For the flags which were defined, it determines if the flag was active or inactive at that time.<br>
For an active flag, it prints the start and end tiem of the segment to which the active flag belongs. <br>
For an inactive flag, it prints the end time of the privious adjacent active segment and the start time of the next adjacent active segment.
</font>
<hr>
</p>

