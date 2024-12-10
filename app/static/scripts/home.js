//Streaming image updates
//TODO: Browser Compatibility     
const img = document.getElementById('uploadedImage');
let imgX = 0;
let imgY = 0;
let zoom = 1;

const startbtn  = document.getElementById("start-btn");
const stopbtn   = document.getElementById("stop-btn");
const logoutbtn = document.getElementById("logout-btn" );
let   stream    = true
let   last_transform = ''

async function fetchData() {
	
	stream = true
	
try {
    const response = await fetch('/video', {
        method: 'GET',
        headers: {
            'Accept': 'text/event-stream;multipart/x-mixed-replace'
        }

    }
    );
    const element        = document.getElementById("uploadedImage");
    const reader         = response.body.getReader();
    const status_elem    = document.getElementById("status");
    const gesture_elem   = document.getElementById("gesture");
    const action_elem    = document.getElementById("action");
    const feedback_elem  = document.getElementById("user-feedback-box");
    const video          = document.getElementById("cam-video-output")
    
    let scale           = 0.0;
    decoder             =  new TextDecoder();
    last_transform      = '';
    let tranform_params = '';
    let buffer          = '';
    let status          = '';
    let gesture         = '';
    let action          = '';
    let boundary        = null
    let frame           = null;
    let jsonStr         = null;

    

    startbtn.setAttribute("hidden", true);
    stopbtn.removeAttribute("hidden");

    //save stream_state  for auto reload
    localStorage.setItem("streamState" , "run");
    
    while (stream) {

        const { done, value } = await reader.read()
        
        if (done) {
            break;
        }
        buffer += decoder.decode(value, {stream:true});
 
        const boundaryIndex = buffer.indexOf('\r\n\r\n')
        
        if (boundaryIndex ){
			
			const boundaryMatch = buffer.match(/--boundary=*.+/g)
			if (boundaryMatch){
				
				boundary = boundaryMatch[1]
	
			}
			
		}
		
		
		// processing the parts 
        while (buffer.indexOf(boundary) !== -1){
			
			const part = buffer.slice(0, buffer.indexOf(boundary))
			buffer     = buffer.slice(buffer.indexOf(boundary) + boundary.length+ 2 ) // + 2 for \n
			
			if (part.includes('Content-Type: application/json')){
				
				   jsonStr = part.slice(part.indexOf('{')).trim()				
				  			
				 }
			else if ( part.includes('Content-Type: image/jpeg\r\n\r\n')){
 
				   
				   frame = part.slice(part.indexOf('Content-Type: image/jpeg\r\n\r\n') + 'Content-Type: image/jpeg\r\n\r\n'.length).trim()	
				   frame = frame.replace('\r\n','')
	
				
				}
								
		}
		
				
        if (jsonStr) {

            jsonData = JSON.parse(jsonStr)
            scale    = jsonData.scale
            pos_x    = jsonData.pos_x
            pos_y    = jsonData.pos_y
            status   = jsonData.status
            gesture  = jsonData.gesture
            action   = jsonData.action
            
            

            transform_params = "translate(" + String(pos_x) + "px," + String(pos_y) + "px) scale(" + String(scale) + ")"
            if (status != 'n/a') {
                status_elem.innerText  = status
                gesture_elem.innerText = gesture
                action_elem.innerText  = action

            }
            
            if (status == 'Active'){
			     feedback_elem.style.backgroundColor = '#28a745' ;
			} else if (status =='Locked'){
				
				feedback_elem.style.backgroundColor =  'silver' ;
				
			}

            if (transform_params != last_transform) {
				console.log(transform_params)
                element.style.transform = transform_params;
                try{ 
					localStorage.setItem("file_transform" , transform_params  );
				}
				catch(error) {
					console.log("transform params error : " + error)
				}
            }
            
            last_transform = transform_params
            
        }
        
                
        
        if (frame){
			
            let url                = "data:image/jpeg;base64," + frame
            video.src              = url
            url                    = null
		}
		
		
		
    }

}

catch (error){
	
	 startbtn.removeAttribute("hidden");
     stopbtn.addAttribute("hidden");
	 console.log("ERROR : " + error)
	
}


}

async function stopData(){ 
	
	  stream = 	false
	 
	
	  try {
          const response = await fetch('/stop_stream', {
                                      method: 'POST',
                                      headers: {
                                            'Accept': '*'
                                          }

                                          }
                          )
                          
          startbtn.removeAttribute("hidden");
          stopbtn.setAttribute("hidden", true);                   
	
	   }catch (error){
	
	      console.log("ERROR : " + error)
	
       }

}

async function lastTransform(){
	
	
	try {
          const response = await fetch('/last_transform', {
                                      method: 'GET',
                                      headers: {
                                            'Accept': 'application/json'
                                          }

                                          }
                          )
                          
        resp = response
        
                          
        if (response.status == 200){
			
	            jsonTransform = resp.json();
	            
	            console.log(jsonTransform);
	       
	       if (jsonTransform){
			   
			   jsonTransform;
			   transform_params = "translate(" + String(jsonTransform.pos_x) + "px," + String(jsonTransform.pos_y) + "px) scale(" + String(jsonTransform.scale) + ")"
			   return transform_params
		   }
			
		}                
	
	   }catch (error){
	
	      console.log("ERROR : " + error)
	
       }

	
	
	
	
}


async function endSession() {
	
                                          try {
                                                   const response = await fetch('/logout', {
                                                    method: 'POST',

                                                    }
                                                  );
                                                  
                                                  console.log(response)
                                                  if (response){
													  const ok  = response.ok
													  const redirected = response.redirected
													  const url = response.url
													  localStorage.clear();
													 if (ok && redirected){
														 
														 window.location.replace(url)														 
														 
													 }
													  
													  
												  }
											  }
										
			                             catch (error){
	
	                                                   error_msg.setAttribute("hidden", false);
	                                                   error_msg.innertText( "ERROR : " + error);
	                                                   console.log("ERROR : " + error);
	
                                                    }
                               }



	


// Function to fetch image state from FastAPI
async function fetchImageState() {
    try {
        const response = await fetch('/image-state');
        const data = await response.json();
        imgX = data.imgX;
        imgY = data.imgY;
        zoom = data.zoom;
        updateImageTransform();  // Apply the new values to the image
    } catch (error) {
        console.error('Error fetching image state:', error);
    }
}


/// save image to local storage for reloads 
function saveToLocalStorage(file, image_state ){
	localStorage.setItem( "file"       , JSON.stringify(file)); 
	//localStorage.setItem( "image_state" ,  image_state); 
	
}


	
	
// Function to handle the image upload
function uploadImage() {
    const fileInput = document.getElementById('imageInput');
    const file = fileInput.files[0];
    
    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            // Set the image src to the file's data URL
            img.src = e.target.result;       
              
            img.style.transform  = "translate(0px,0px) scale(1)"		     
            img.style.display = 'block';  // Make sure the image is visible
            //saving file in local storage//
            saveToLocalStorage(reader.result)
        };
        reader.readAsDataURL(file);  // Read the file as a data URL
    }

}

// load the last image on page reload //
async function loadCachedImage(){
	    console.log("loading cached image")
		const storedImageSrc = localStorage.getItem("file")
		if (storedImageSrc){
			
		try{  		  
		   console.log("transforming image")
		   img_transform = localStorage.getItem("file_transform");
		   //console.log(img_transform );
		   //TODO

		   console.log(img_transform );
		   
		     
		   img.src = JSON.parse(storedImageSrc) 
           img.style.display = 'block'; 
           ing.style.transform = img_transform;
           
		    } catch(error) {
					console.log("transform img error : " + error);
				 }
		    
			
	   }
				
}


async function clearImage(){
	
	img.src = "";
	img.alt = "";
	img.removeAttribute("src");
	localStorage.clear();
	
}


// Function to move image and update state
async function moveImage(direction) {
    const moveStep = 20;
    switch (direction) {
        case 'up':
            imgY -= moveStep;
            break;
        case 'down':
            imgY += moveStep;
            break;
        case 'left':
            imgX -= moveStep;
            break;
        case 'right':
            imgX += moveStep;
            break;
    }
    await updateImageState();  // Send updated values to FastAPI
    updateImageTransform();    // Update image on the screen
}

// Function to zoom in or out and update state
async function zoomImage(type) {
    const zoomStep = 0.1;
    if (type === 'in') {
        zoom += zoomStep;
    } else if (type === 'out') {
        zoom -= zoomStep;
    }
    await updateImageState();  // Send updated values to FastAPI
    updateImageTransform();    // Update image on the screen
}

// Function to send updated image state to FastAPI
async function updateImageState() {
    try {
        await fetch('/update-image-state', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ x: imgX, y: imgY, zoom: zoom }),
        });
    } catch (error) {
        console.error('Error updating image state:', error);
    }
}

// Function to apply image position and zoom from state
function updateImageTransform() {
    img.style.transform = `translate(${imgX}px, ${imgY}px) scale(${zoom})`;
}



//Listeners
startbtn.addEventListener("click", () => {

    fetchData();
 
});

stopbtn.addEventListener("click", () => {
	
	//save stream state  for subsequent page loads
    localStorage.setItem("streamState" , "stop");
    stopData();
 
});

logoutbtn.addEventListener("click", () => {
	stopData();
	localStorage.clear();
	endSession();
});

window.addEventListener("beforeunload" , () => { 

	stopData();
	
	});
	
window.addEventListener("load" , () => { 
	
	//check stream state
	stream_state = localStorage.getItem("streamState")
    stopData();
	// load cached image  //
	console.log("Loading Image");
	
	loadCachedImage();

	if (stream_state == "run"){
	      fetchData();
    }
	});
    
 // Fetch the image state initially when the page loads
window.onload = fetchImageState; 