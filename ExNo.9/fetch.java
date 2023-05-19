import java.io.*;
import java.util.*;
import javax.servlet.*;
import javax.servlet.http.*;
import java.sql.*;
import java.sql.DriverManager;
import java.sql.Connection;
import java.sql.SQLException;
public class fetch extends HttpServlet{
    public void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
        response.setContentType("text/html");
        Connection conn=null;
        PreparedStatement stmt=null;
        PrintWriter out = response.getWriter();


        String email = request.getParameter("email");
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
            sql = "SELECT * FROM details where email = ?";
            stmt = conn.prepareStatement(sql);
            stmt.setString(1, email);


            ResultSet rs = stmt.executeQuery();
            // Extract data from result set
            while(rs.next())
            {
                //out.println("<h1> into while loop</h1>");
                //Retrieve by column name
                String name = rs.getString("name");
                String contact = rs.getString("contact");
                String address = rs.getString("address");
               
                //Display values
                out.println("Name: " + name + "<br>");
                out.println("Email: " + email + "<br>");
                out.println("Contact: " + contact + "<br>");
                out.println("Address: " + address + "<br>");
   
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