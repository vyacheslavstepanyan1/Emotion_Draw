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
  const generate = async (prompt) => {
    setLoading(true);
    const response = await axios.get(`http://127.0.0.1:8000/?prompt=${prompt}`);
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
                      label={<Box m={2}>Hi there, I'm EmoBERT!<br />
                      I'm the language model ALBERT with a twist.<br />
                      These enthusiastic data scientists have fine-tuned me to be extra emotional.<br />
                      Click on my tummy to learn more about me.<br />
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
                      label={<Box color='white' m={2}>Hello, I'm Stable Diffusion!<br />
                      You can also call me Diffusio 🥰<br />
                      I'm a text-to-image model.<br />
                      That quirky monster on the left makes me create unusual images.<br />
                      Click on my tummy to find out more about me.<br />
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
                    label={<Box m={2}>Welcome to our project!<br />
                    We've accidentally fine-tuned the ALBERT model, and now it’s super emotional! 🫣<br />
                    We call this expressive version EmoBERT because it breaks down every sentence into three distinct emotions.<br />
                    EmoBERT has a creative companion named Diffusion (or Diffusio).<br />
                    Diffusio loves to draw, especially for EmoBERT.<br />
                    Share a sentence with EmoBERT, and Diffusio will illustrate the emotions EmoBERT discovers.</Box>}
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