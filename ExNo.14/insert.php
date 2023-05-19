<?php
    $conn = new mysqli("localhost", "root", "", "details");
    if(isset($_POST['submit'])){ // Fetching variables of the form which travels in URL
        
        $name = $_POST['name'];
        $email = $_POST['email'];
        $mobile = $_POST['mobile'];
        $password = $_POST['password'];
        $cpassword = $_POST['cpassword'];
        
        if($name !=''||$email !=''){
        //Insert Query of SQL
            $sql = "insert into detail(name, email, mobile, password, cpassword) values ('$name', '$email', '$mobile', '$password', '$cpassword')";
            if ($conn->query($sql) === TRUE) {
                echo "record inserted successfully";
            } else {
                echo "Error: " . $sql . "<br>" . $conn->error;
            }
        }
        else{
        echo "<p>Insertion Failed <br/> Some Fields are Blank....!!</p>";
        }
    }
    $conn->close(); // Closing Connection with Server
?>