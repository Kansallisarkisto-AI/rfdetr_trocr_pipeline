from bs4 import BeautifulSoup as bs
from yattag import indent
from pathlib import Path
import os
from lxml import etree
import datetime

class PageXML:

    def save_page(self, xml_file, path):
        """Saves Page xml file."""
        xml_file.write(path, xml_declaration=True, encoding='utf-8', method="xml")
        #print('Page XML file saved to ', path)

    def format_polygon(self, polygon):
        """Formats polygon from a list of lists into a string."""
        polygon_str = ''
        for pair in polygon:
            polygon_str += '%s,%s '%(int(pair[0]), int(pair[1]))
        return polygon_str.rstrip()

    def create_xml(self, data, image_path): 
        attr_qname = etree.QName("http://www.w3.org/2001/XMLSchema-instance", "schemaLocation")
        root = etree.Element('PcGts',
                            nsmap={None: 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'})

        # Metadata
        m = etree.SubElement(root, "Metadata")
        cr = etree.SubElement(m, "Creator")
        cr.text = 'Kansallisarkisto - National Archives of Finland'

        now = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f+02:00')
        cd = etree.SubElement(m, "Created")
        cd.text = now
        lc = etree.SubElement(m,"LastChange")
        lc.text = now

        # Root -> Page
        p = etree.SubElement(root, "Page")
        p.set('imageFilename', data[0]['img_name'])
        p.set('imageWidth', str(data[0]['width']))
        p.set('imageHeight', str(data[0]['height']))
        
        # Loop over regions
        for i, region_dict in enumerate(data):
            region_id = 'r' + str(i)
            region_name = region_dict['region_name']
            region_polygon_coords = self.format_polygon(region_dict['region_coords'])
            custom_region_tag = "readingOrder {index:%s;}" % str(i)

            # Page -> TextRegion
            tr = etree.SubElement(p, "TextRegion")
            tr.set('id', region_id)
            tr.set('custom', custom_region_tag)
            
            # TextRegion -> Coords
            region_coords = etree.SubElement(tr,"Coords")
            region_coords.set('points', region_polygon_coords)
            
            # Loop over text lines belonging to the region
            for j, line_dict in enumerate(region_dict['text_lines']):
                line_id = region_id + 'l' + str(j)
                line_text = line_dict['text']
                line_polygon_coords = self.format_polygon(line_dict['polygon'])
                custom_line_tag = "readingOrder {index:%s;}" % str(j)

                # TextRegion -> TextLine
                tl = etree.SubElement(tr, "TextLine")
                tl.set('id', line_id)
                tl.set('custom', custom_line_tag)
                
                # TextLine -> Coords
                line_coords = etree.SubElement(tl, "Coords")
                line_coords.set('points', line_polygon_coords)

                # Add Baseline element
                baseline = etree.SubElement(tl, "Baseline")
                baseline.set('points', '')

                # TextLine -> TextEquiv
                te = etree.SubElement(tl, "TextEquiv")
                # TextEquiv -> Unicode
                uc = etree.SubElement(te, "Unicode")
                uc.text = line_text.strip()
                
        xml_doc = etree.ElementTree(root)

        return xml_doc
    
    def get_page(self, page_dict, image_path, save_path):
        page = self.create_xml(page_dict, image_path)
        self.save_page(page, save_path)


class AltoXML:
    def __init__(self, seg_model, line_model, htr_model):
        self.seg_model = seg_model
        self.line_model = line_model
        self.htr_model = htr_model
    
    def save_alto(self, newsoup, path):
        """Saves Alto xml file."""
        with open(path,"w") as f: 
            f.write(indent(str(newsoup))) 
        print('XML file saved to ', path)

    def format_polygon(self, polygon):
        """Formats polygon from a list of lists into a string."""
        polygon_str = ''
        for pair in polygon:
            polygon_str += '%s,%s '%(int(pair[0]), int(pair[1]))
        return polygon_str
    
    def get_region_ids(self, data):
        """Creates region ids based on the region names."""
        region_names = [region_dict['region_name'] for region_dict in data]
        unique_names = list(set(region_names))
        ind_dict = {name: 0 for name in unique_names}
        new_names = []
        for i, name in enumerate(region_names):
            ind = ind_dict[name]
            new_name = name + '_' + str(ind)
            new_names.append(new_name)
            ind_dict[name] = ind_dict.get(name, 0) + 1
        return new_names

    def get_min_max_coordinates(self, data):
        all_polygons = []
        for region_dict in data:
            for line_dict in region_dict['text_lines']:
                int_polygon = [[int(x), int(y)] for x, y in line_dict['polygon']]
                all_polygons.append(int_polygon)

        all_points = [point for sublist in all_polygons for point in sublist]

        x_values = [p[0] for p in all_points]
        y_values = [p[1] for p in all_points]

        min_x, max_x = min(x_values), max(x_values)
        min_y, max_y = min(y_values), max(y_values)

        return min_x, max_x, min_y, max_y
    
    def get_polygon_min_max_coordinates(self, polygon_coords):
        int_polygon = [[int(x), int(y)] for x, y in polygon_coords]

        x_values = [p[0] for p in int_polygon]
        y_values = [p[1] for p in int_polygon]

        min_x, max_x = min(x_values), max(x_values)
        min_y, max_y = min(y_values), max(y_values)

        return min_x, max_x, min_y, max_y

    def create_xml(self, data, image_path):
        """
        Serializes text line polygons and predicted text content into ALTO XML format.

        Constructs an ALTO-compliant XML document from the processed predictions,
        encoding page layout information (regions, text lines, coordinates) alongside
        OCR content and confidence metrics for each region.

        Parameters
        ----------
        data : list[dict]
            A list of region dictionaries as returned by `process_text_predictions`.
            Each dict contains:
                - img_name        (str)   : Source image filename.
                - height          (int)   : Page image height in pixels.
                - width           (int)   : Page image width in pixels.
                - page_conf_mean  (float) : Mean OCR confidence across the page.
                - page_conf_median(float) : Median OCR confidence across the page.
                - page_conf_25    (float) : 25th percentile OCR confidence.
                - page_conf_75    (float) : 75th percentile OCR confidence.
                - n_long_rowtext  (int)   : Number of long row-text lines on the page.
                - language        (str)   : Detected or specified language of the page.
                - region_conf     (float) : Confidence score for this region's segmentation.
                - region_coords   (list)  : Polygon coordinates defining the region boundary.
                - region_name     (str)   : Label or class name of the region.
                - text_lines      (list[dict]) : Per-line dicts with polygon coordinates
                                                and predicted text content. Dicts contain 'text', 
                                                'row_length', 'text_conf, 'polygon' keys

        image_path : str or Path
            Filesystem path to the source image file. Written into the ALTO XML
            as the source file reference.
        seg_model_arch : {'yolo', 'rfdetr'}, optional
            Architecture of the segmentation model used to produce the layout predictions.
            Determines which model name is recorded in the ALTO XML processing metadata.
            Defaults to 'rfdetr'.

        Returns
        -------
        xml.etree.ElementTree.ElementTree (or str)
            The constructed ALTO XML document.

        """
        # xml template where to start building the Alto xml
        newsoup=bs(f"""
        <alto xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
        xmlns="http://www.loc.gov/standards/alto/ns-v4#"
        xsi:schemaLocation="http://www.loc.gov/standards/alto/ns-v4# http://www.loc.gov/standards/alto/v4/alto-4-0.xsd"
        xmlns:xlink="http://www.w3.org/1999/xlink">
        <Description>
            <MeasurementUnit>mm10</MeasurementUnit>
            <sourceImageInformation>
                <fileName>{image_path}</fileName>
            </sourceImageInformation>
            <Processing ID="TEXT_REGION_SEGMENTATION">
                <processingAgency>Kansallisarkisto - National Archives of Finland</processingAgency>
                <processingStepDescription>Text region detection</processingStepDescription>
                <processingSoftware>
                    <softwareCreator>Roboflow</softwareCreator>
                    <softwareName>RFDETRSegPreview</softwareName>
                    <softwareVersion>{self.seg_model}</softwareVersion>
                </processingSoftware>
            </Processing>
            <Processing ID="TEXT_LINE_SEGMENTATION">
                <processingAgency>Kansallisarkisto - National Archives of Finland</processingAgency>
                <processingStepDescription>Text line detection</processingStepDescription>
                <processingSoftware>
                    <softwareCreator>Roboflow</softwareCreator>
                    <softwareName>RFDETRSegPreview</softwareName>
                    <softwareVersion>{self.line_model}</softwareVersion>
                </processingSoftware>
            </Processing>
            <Processing ID="TEXT_RECOGNITION">
                <processingAgency>Kansallisarkisto - National Archives of Finland</processingAgency>
                <processingStepDescription>Handwritten text recognition</processingStepDescription>
                <processingSoftware>
                    <softwareCreator>Microsoft</softwareCreator>
                    <softwareName>TrOCR</softwareName>
                    <softwareVersion>{self.htr_model}</softwareVersion>
                </processingSoftware>
            </Processing>
        </Description>
        <Styles></Styles>
        <Layout>        
        </Layout>
        </alto>
        ""","xml")
        
        # create Page element
        new_tag_page=newsoup.new_tag("Page", 
                ID=data[0]['img_name'],
                WIDTH=data[0]['width'],
                HEIGHT=data[0]['height'],
                PC="{0:.2f}".format(data[0]['page_conf_mean']),
                PC50="{0:.2f}".format(data[0]['page_conf_median']),
                PC25="{0:.2f}".format(data[0]['page_conf_25']),
                PC75="{0:.2f}".format(data[0]['page_conf_75']),
                N_LONG_ROWTEXT=data[0]['n_long_rowtext'],
                LANGUAGE=data[0]['language'],
                PHYSICAL_IMG_NR=0)
        
        # calculate min and max xs and ys
        min_x, max_x, min_y, max_y = self.get_min_max_coordinates(data)

        # Add margins
        topmargin_tag = newsoup.new_tag("TopMargin", 
                                    HPOS="0",
                                    VPOS="0",
                                    WIDTH=str(data[0]['width']),
                                    HEIGHT=str(int(min_y)),
                                    ID="TM_00001")
        new_tag_page.append(topmargin_tag)

        leftmargin_tag = newsoup.new_tag("LeftMargin", 
                                    HPOS="0",
                                    VPOS=str(int(min_y)),
                                    WIDTH=str(int(min_x)),
                                    HEIGHT=str(int(max_y)-int(min_y)),
                                    ID="LM_00001")
        new_tag_page.append(leftmargin_tag)

        rightmargin_tag = newsoup.new_tag("RightMargin", 
                                    HPOS=str(int(max_x)),
                                    VPOS=str(int(min_y)),
                                    WIDTH=str(int(data[0]['width'])-int(max_x)),
                                    HEIGHT=str(int(max_y)-int(min_y)),
                                    ID="RM_00001")
        new_tag_page.append(rightmargin_tag)

        bottommargin_tag = newsoup.new_tag('BottomMargin',
                                    HPOS="0",
                                    VPOS=str(int(max_y)),
                                    WIDTH=str(int(data[0]['width'])),
                                    HEIGHT=str(int(data[0]['height'])-int(max_y)),
                                    ID="BM_00001")
        new_tag_page.append(bottommargin_tag)
        
        new_tag_printspace=newsoup.new_tag("PrintSpace", 
                                    HPOS=str(int(min_x)),
                                    VPOS=str(int(min_y)),
                                    WIDTH=str(int(max_x)-int(min_x)),
                                    HEIGHT=str(int(max_y)-int(min_y)),
                                    ID="PS_00001")
        
        new_tag_page.append(new_tag_printspace)    
        newsoup.Layout.append(new_tag_page)
        
        # add elements to the page
        composed_blocks=bs('',"html.parser")
        region_ids = self.get_region_ids(data)
        # Loop over detected regions
        for i, region_dict in enumerate(data):
            # Calculate HPOS, VPOS, WIDTH and HEIGHT for textblock
            min_x, max_x, min_y, max_y = self.get_polygon_min_max_coordinates(region_dict['region_coords'])

            # Create ComposedBlock
            composed_block = newsoup.new_tag("ComposedBlock", 
                                    HPOS=str(int(min_x)),
                                    VPOS=str(int(min_y)),
                                    WIDTH=str(int(max_x)-int(min_x)),
                                    HEIGHT=str(int(max_y)-int(min_y)),
                                    ID="block_1_" + str(i+1))

            # Create TextBlock element for each region
            text_block=newsoup.new_tag("TextBlock", 
                                    HPOS=str(int(min_x)),
                                    VPOS=str(int(min_y)),
                                    WIDTH=str(int(max_x)-int(min_x)),
                                    HEIGHT=str(int(max_y)-int(min_y)),
                                    ID=region_ids[i])

            # Add text region polygon values to Shape tag of the text block
            region_shape=newsoup.new_tag("Shape")
            region_polygon_str = self.format_polygon(region_dict['region_coords'])
            region_polygon = newsoup.new_tag("Polygon", POINTS=region_polygon_str)
            region_shape.append(region_polygon)
            text_block.append(region_shape)
            # Loop over text lines belonging to the region
            for j, line_dict in enumerate(region_dict['text_lines']):
                line_id = region_ids[i] + '_line_' + str(j)

                # Calculate HPOS VPOS WIDTH and HEIGHT for text line
                min_x, max_x, min_y, max_y = self.get_polygon_min_max_coordinates(line_dict['polygon'])

                # Create TextLine element for each detected text line
                text_line=newsoup.new_tag("TextLine", 
                                            HPOS=str(int(min_x)),
                                            VPOS=str(int(min_y)),
                                            WIDTH=str(int(max_x)-int(min_x)),
                                            HEIGHT=str(int(max_y)-int(min_y)),
                                            ID=line_id) 
                text_string=newsoup.new_tag("String", 
                                            HPOS=str(int(min_x)),
                                            VPOS=str(int(min_y)),
                                            WIDTH=str(int(max_x)-int(min_x)),
                                            HEIGHT=str(int(max_y)-int(min_y)),
                                            ID=line_id, 
                                            CONTENT=line_dict['text'], 
                                            RL=line_dict['row_length'], 
                                            WC=str(round(line_dict['text_conf'],2)))
                # Add text line polygon values to Shape tag of the text line
                line_shape=newsoup.new_tag("Shape")
                line_polygon_str = self.format_polygon(line_dict['polygon'])
                line_polygon = newsoup.new_tag("Polygon", POINTS=line_polygon_str)
                line_shape.append(line_polygon)
                text_line.append(line_shape)
                text_line.append(text_string)
                text_block.append(text_line)
            
            composed_block.append(text_block)
            composed_blocks.append(composed_block)
        
        newsoup.PrintSpace.append(composed_blocks)

        return newsoup

    def get_alto(self, page_dict, image_path, save_path):
        alto = self.create_xml(page_dict, image_path)
        self.save_alto(alto, save_path)


def get_xml(text_predictions, input_data):
    """Function for saving the results in xml file.
    
    Save OCR results to XML file in PAGE and/or ALTO format.

    Args:
        text_predictions: The OCR text predictions to be saved.
        input_data: Object containing configuration parameters including:
            - image_path: Path to the input image file.
            - page_xml: Boolean flag to enable PAGE XML output.
            - alto_xml: Boolean flag to enable ALTO XML output.
            - xml_path: Base directory path for saving XML files.
            - region_segment_model: Model used for region segmentation (for ALTO).
            - line_segment_model: Model used for line segmentation (for ALTO).
            - text_recognition_model: Model used for text recognition (for ALTO).

    Returns:
        None. XML files are saved to disk in subdirectories ('page' and/or 'alto')
        within the specified xml_path.
    """
    xml_name = Path(input_data.image_path).stem + '.xml'
    # PAGE XML option
    if input_data.page_xml and input_data.xml_path:
        page_maker = PageXML()
        save_folder = Path(input_data.xml_path, 'page')
        os.makedirs(save_folder, exist_ok=True)
        save_path = str(Path(save_folder, xml_name))
        page_maker.get_page(text_predictions, input_data.image_path, save_path)
    # Alto XML option
    if input_data.alto_xml and input_data.xml_path:
        save_folder = Path(input_data.xml_path, 'alto')
        os.makedirs(save_folder, exist_ok=True)
        save_path = str(Path(save_folder, xml_name))
        alto_maker = AltoXML(input_data.region_segment_model_name, input_data.line_segment_model_name, input_data.text_recognition_model_name)
        alto_maker.get_alto(text_predictions, input_data.image_path, save_path)
