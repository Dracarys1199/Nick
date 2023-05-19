import java.io.*;
import java.util.*;
import javax.servlet.*;
import javax.servlet.http.*;
import java.sql.*;


public class fetchAll extends HttpServlet{
    public void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        response.setContentType("text/html");
        Connection conn=null;
        Statement stmt=null;
        PrintWriter out = response.getWriter();


        try
        {
            Class.forName("com.mysql.jdbc.Driver");
            //out.println("<h1> into class</h1>");
            //create a database connection using jdbc , port no used here is 3307
            // database name is college and username is root , there is no password
            conn = DriverManager.getConnection("jdbc:mysql://localhost:3306/aai", "root", "");
            if(conn!=null)
            {
                out.println("<h1> No issues in Connection</h1>");
            }
            String sql;
            //display all the students' marks
            stmt = conn.createStatement();
            sql = "SELECT * FROM details";
            ResultSet rs = stmt.executeQuery(sql);


            // Extract data from result set
            out.println("<table cellspacing='0' width='350px' border='1'>");
            out.println("<tr>");
            out.println("<td> Name </td>");
            out.println("<td> Email </td>");
            out.println("<td> Contact </td>");
            out.println("<td> Address </td>");
           
            out.println("</tr>");
           
            while(rs.next())
            {
             out.println("<tr>");
             out.println("<td>"  + rs.getString("name")   +  "</td>");
             out.println("<td>"  + rs.getString("email")   +  "</td>");  
             out.println("<td>"  + rs.getString("contact")   +  "</td>");  
             out.println("<td>"  + rs.getString("address")   +  "</td>");  
           
             out.println("</tr>");
 
            }
            out.println("</body></html>");
            // Clean-up environment
            rs.close();
            stmt.close();
            conn.close();
        }
        catch(Exception e)
        {
            System.out.print("Do not connect to DB - Error:"+e);
        }
    }
}