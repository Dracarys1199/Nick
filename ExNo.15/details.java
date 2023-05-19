/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package monica;

import javax.jws.WebService;
import javax.jws.WebMethod;
import javax.jws.WebParam;
import java.sql.*;
import java.sql.DriverManager;
import java.sql.Connection;
import java.sql.SQLException;

/**
 *
 * @author monica
 */
@WebService(serviceName = "Details")
public class details {

    /**
     * This is a sample web service operation
     */
    String Result = "No records found :((";
    @WebMethod(operationName = "name")
public String hello(@WebParam(name = "name") String name) {
        

   try
   {
       Class.forName("com.mysql.jdbc.Driver");
       try (Connection conn = DriverManager.getConnection("jdbc:mysql://127.0.0.1:3306/details","root", "")) {
           if(conn!=null)
           {
               System.err.println("<h1> No issues in Connection</h1>");
           }
           try (Statement stmt = conn.createStatement()) {
               String sql;
               sql = "SELECT name,email FROM details WHERE name='" + name + "'" ;
               System.out.println("Name: ");
               // Extract data from result set
               try (ResultSet rs = stmt.executeQuery(sql)) {
                   // Extract data from result set
                   while(rs.next())
                   {
                       //Retrieve by column name
                       String sname = rs.getString("name");
                       String mail = rs.getString("email");
                       System.out.println("Name: " + sname);
                       
                       Result = "<User><name>" + sname + "</name><Mail>" + mail + "</Mail></User>";

                       System.err.format("%s,%s\n",sname, mail);
                   }
                   if (Result.equals("No records found :((")) {
                       System.out.println("No records found for customer ID: " + name);
                   }
               }
           }
       }
   }
   catch(ClassNotFoundException | SQLException e)
   {
       System.err.print("Do not connect to DB - Error:"+e);
   }
        
   return Result;
}


}