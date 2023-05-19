/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package monica;

import java.io.IOException;
import java.io.PrintWriter;
import javax.servlet.ServletException;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

public class fetch extends HttpServlet {

    protected void doPost(HttpServletRequest request, HttpServletResponse response)
            throws ServletException, IOException {

        details detail = new details();
        response.setContentType("text/html;charset=UTF-8");
        PrintWriter out = response.getWriter();

        String name = request.getParameter("name");
        String xmlResult = detail.hello(name);

        // Parse XML response and display results in table rows
        out.println("<html>");
        out.println("<head><title>User Details</title></head>");
        out.println("<body>");
        out.println("<table border=\"1\">");
        out.println("<tr><th>Name</th><th>Email</th></tr>");
        
        String mail = "", sname = "";
        int startIndex, endIndex;
        while (xmlResult.contains("<User>")) {
            startIndex = xmlResult.indexOf("<name>") + 6;
            endIndex = xmlResult.indexOf("</name>");
            sname = xmlResult.substring(startIndex, endIndex);
            xmlResult = xmlResult.substring(endIndex + 5);
            
            startIndex = xmlResult.indexOf("<Mail>") + 6;
            endIndex = xmlResult.indexOf("</Mail>");
            mail = xmlResult.substring(startIndex, endIndex);
            xmlResult = xmlResult.substring(endIndex + 7);
            
            out.println("<tr><td>" + sname + "</td><td>" + mail + "</td></tr>");
        }
        
        out.println("</table>");
        out.println("</body></html>");
    }
}