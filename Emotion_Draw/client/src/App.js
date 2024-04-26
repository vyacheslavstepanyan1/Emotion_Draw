import {
  ChakraProvider,
  Heading,
  Container,
  Text,
  Input,
  Button,
  Wrap,
  Stack,
  Image,
  Link,
  SkeletonCircle,
  SkeletonText,
  Box,
  useDisclosure, 
  Modal, 
  ModalOverlay, 
  ModalContent, 
  ModalHeader, 
  ModalCloseButton, 
  ModalBody,Flex, 
  Tooltip,
  Progress, 
  Spacer, 
  Square
} from "@chakra-ui/react";
import axios from "axios";
import { useState, useRef } from "react";
import videoBG from "./assets/videoplayback.webm";
import berto from "./assets/bert_monster.png";
import difuso from "./assets/difus_monster.png";
import "./index.css";

const App = () => {
  const [images, setImages] = useState([]); 
  const [emotions, setEmotions] = useState([]); 
  const [prompt, setPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const { isOpen, onOpen, onClose } = useDisclosure();
  const [selectedImage, setSelectedImage] = useState(null); 
  const [isImageModalOpen, setImageModalOpen] = useState(false);
  const handleImageDownload = () => {
    const link = document.createElement('a');
    link.href = selectedImage; 
    link.download = "downloaded-image.png"; 
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };
  const handleImageClick = (image) => {
    setSelectedImage(image);
    setImageModalOpen(true);
  };
  const config = {
    headers: {
      'ngrok-skip-browser-warning': 'true'
    }
  };  
  const generate = async (prompt) => {
    setLoading(true);
    const response = await axios.get(`https://306f-2a02-2a57-7161-0-2593-6cfc-a622-6cf0.ngrok-free.app/?prompt=${prompt}`, config);
    if (response.data && Array.isArray(response.data)) {
      const imageUrls = response.data.filter(item => item.media_type === "image/png").map(img => `data:image/png;base64,${img.body}`);
      setImages(imageUrls);
      const emotionTexts = response.data.filter(item => typeof item === "string");
      setEmotions(emotionTexts);
    }
    setLoading(false);
  };
  const resetAppState = () => {
    setImages([]);
    setEmotions([]);
    setPrompt("");
  };
  const inputRef = useRef(null);
  const handleKeyPress = (event) => {
    if (event.key === 'Enter' && inputRef.current) {
      generate(prompt);
      inputRef.current.blur();
    }
  };

  return (
    <div className="main">
      <div className="overlay"></div>
      <video src={videoBG} autoPlay loop muted />
      <div className="content">
        <ChakraProvider>
          <Flex direction="column" align="center" justify="center" className="main">  
            <Tooltip bg="rgb(61, 232, 255, .2)"
                      label={<Box m ={2} >My name is AlBert.<br />
                      I am a language model.<br />
                      These crazy data scientists fine tunned me.<br />
                      Click my tummy to get more information.<br />
                      </Box>}
                      placement="right" hasArrow right = '550'>
              <Link href="https://huggingface.co/docs/transformers/en/model_doc/albert"isExternal> {/* Make each image a clickable link */}
                <Image
                  src={berto}
                  display={loading || images.length !== 0 ? "none": "block"}
                  position="absolute"
                  left="10"
                  top="50%"
                  transform="translateY(-50%) scale(0.8)"
                />
              </Link>
            </Tooltip>
            
            <Tooltip bg = 'rgb(255, 150, 194, .2)'
                      label={<Box color='white' m ={2}>My name is Stable Diffusion.<br />
                      I am a Text to Image model.<br />
                      This Monster on the left makes me draw wiered images.<br />
                      Click my tummy to get more information.<br />
                      </Box>}
                      placement="left" hasArrow left = '550'>
              <Link href="https://huggingface.co/CompVis/stable-diffusion-v1-4"isExternal> 
                <Image
                  src={difuso}
                  display={loading || images.length !== 0 ? "none": "block"}
                  position="absolute"
                  right="10"
                  top="50%"
                  transform="translateY(-50%) scale(0.8)"
                />
              </Link>
                
            </Tooltip>
              
            <Container maxW="100%">
              <Flex as="header" width="100%" align="center" justify="center" p={4} position="fixed" top="20" zIndex="banner">
                <Heading size="4xl" textAlign="center">
                  <Link href="https://www.linkedin.com/in/vyacheslavstepanyan/" isExternal _hover={{
                  textDecoration: 'none'}}>🐰</Link>
                  Emotion Draw
                  <Link href="https://www.linkedin.com/in/anahitabaghdasaryan/" isExternal _hover={{
                  textDecoration: 'none'}}>🐭</Link>
                  <Tooltip 
                    label={<Box m ={2}>This is our project. We are muk and zaich<br />
                    We accidentaly finetuned a bert model and now it is working!<br />
                    Our Bert is very emotional Bert, so it always breaks any sentence to 3 emotions.<br />
                    EmoBert has a friend. Friends' name is Diffusion a.k.a. Difuzyor<br />
                    Difuzyor loves to draw, especially for EmoBert.<br />
                    Tell something to our EmoBert, and Difuzyor will draw the emotions that Emo found.</Box>}
                    placement="right">
                    <Button onClick={onOpen} p={0} background="transparent" _hover={{ background: "transparent" }}>
                      <Image
                        src="https://wikis.tid.es/gvp-public/images/9/9f/Infobox_info_icon_white.svg.png"
                        alt="Info Icon"
                        boxSize="24px"
                      />
                    </Button>
                  </Tooltip>
                  </Heading>
                </Flex>

              <Modal isOpen={isImageModalOpen} onClose={() => setImageModalOpen(false)} size="5xl" isCentered>
                <ModalOverlay bg="rgba(0, 0, 0, 0.8)" />
                <ModalContent bg="rgba(255, 255, 255, 0.2)" borderRadius="lg">
                  <ModalCloseButton />
                  <ModalBody>
                    <Flex direction="column" h="100%" my="8">
                      <Flex flex="1" alignItems="center" justifyContent="center">
                        <Square>
                          <Image src={selectedImage} boxSize='800px' />
                        </Square>
                      </Flex>
                      <Flex justifyContent="flex-end">
                        <Spacer />
                        <Button position="absolute" bottom="0" right="0" colorScheme="blue" background="transparent" _hover={{ background: "transparent" }} onClick={handleImageDownload} mb = "8" mr = "8" scaleX= "2">
                          <Image
                          src="https://iihl.org/wp-content/uploads/2020/02/download-icon-white-png-1.png"
                          alt="Download icon"
                          boxSize="34px"
                          transform="scaleX(1.5)"
                          transition="transform 0.2s"
                          _hover={{
                            transform: 'scale(1.2) scaleX(1.5)' 
                          }}
                          />
                        </Button>
                      </Flex>
                    </Flex>
                  </ModalBody>
                </ModalContent>
              </Modal>

              <Wrap justify="center" marginBottom={"10px"}>
                <Input
                  ref={inputRef}
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="Tell us what is on your mind 👀"
                  width={"350px"}
                  backgroundColor="rgba(0, 0, 0, 0.1)"
                  onKeyDown={handleKeyPress}
                  sx={{
                    '::placeholder': {
                      color: 'rgb(200, 200, 200)',
                    },
                    border: "none",  
                    boxShadow: "none"
                  }}

                />
                <Button
                      onClick={() => {
                        if (inputRef.current) {
                          generate(prompt);
                          inputRef.current.blur();
                        }
                      }}
                      colorScheme={"purple"}
                    >
                      Send
                </Button>
              </Wrap>
              
              {loading && 
              <Box padding='20' boxShadow='xl' width="1200px" marginX="auto" borderRadius="20px">
                <SkeletonCircle size='10'startColor='pink.100' endColor='white.100'/>
                <SkeletonText mt='4' noOfLines={4} spacing='4' skeletonHeight='2' startColor='pink.100' endColor='white.100'/>
              </Box>}

              {!loading && images.length > 0 && (
                <>
                  <Stack direction="row" spacing={20} justify="center">
                      {images.map((src, index) => (
                        <Box key={index} boxShadow="lg" borderRadius="20px" textAlign="center" mt="10" cursor="pointer">
                          <Image
                            borderRadius="20px"
                            src={src}
                            alt={`Generated image ${index + 1}`}
                            maxW="400px"
                            maxH="400px"
                            objectFit="cover"
                            onClick={() => handleImageClick(src)}
                            sx={{
                              transition: 'transform 0.5s ease-in-out',
                              _hover: {transform: 'scale(1.1)'}
                            }}
                          />
                          <Text mt="10px" mb="10px">{emotions[index]}</Text>
                        </Box>
                      ))}
                  </Stack>
                    <Flex justifyContent="center" mt="4">
                    <Button colorScheme="purple" onClick={resetAppState}> Reset</Button>
                    </Flex>
                </>
              )}
            </Container>
          </Flex>
        </ChakraProvider>
      </div>
    </div>
  );
};

export default App;