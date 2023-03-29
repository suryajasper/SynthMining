import React from 'react';
import '../css/tags.scss';
import { fetchRequest, initArr } from '../utils/utils';
import { icons } from './icons';
import { TagAttrs } from './project-loader';
import { PopupManagerState, withPopupStore } from './hooks/popup-state';

interface CategoryViewAttrs {
  isAdmin: boolean;
  tagName: TagAttrs;
  selected: boolean;
  highlighted: number;
  color: string;
  applyToImageMode: boolean;

  select: () => void;
  applyToImage: () => void;
  activatePopup: () => void;
}

function CategoryView(props: CategoryViewAttrs) {
  const tag = props.tagName;

  return (
    <div
      className={[
        'category-item',
        props.selected ? 'selected show-category-description' : '',
        props.highlighted ? 'show-category-description' : '',
      ].join(' ')}

      style={ {
        '--highlight-fill': props.highlighted,
      } as React.CSSProperties }

      onClick={() => {
        if (props.applyToImageMode)
          props.applyToImage();
        else
          props.select();
      }}
    >
      <div className='category-header'>
        <span className='category-item-color' style={{ backgroundColor: props.color }} />
        <span className='category-item-title'>{tag.name}</span>
      </div>

      <div className='category-description'>{tag.description}</div>

      { (props.isAdmin && !props.applyToImageMode) && 
        <div className='hover-menu'>
          <button className='tag-edit-button'
            onClick={e => {
              e.stopPropagation();
              props.activatePopup();
            }}
          >
            {icons.gear}
          </button>
        </div>
      }
    </div>
  );
};

interface CategorySelectionsAttrs {
  uid: string | undefined;
  isAdmin: boolean;
  projectId: string;

  categories: TagAttrs[];
  activeCategories: Record<string, number>;

  store: PopupManagerState;

  addTag: () => void;
  updateTag: (tagId: string, update: {
    name: string,
    description: string,
    goalQty: number,
  }) => void;
  removeTag: (tagId: string) => void;

  // updatePopupStatus: (state: {
  //   name: string,
  //   active: boolean,
  //   data: object,
  // }) => void;
  unshiftSelection: (i: number) => void;
  applyToImage: (tag: TagAttrs) => void;

  applyToImageMode: boolean;
}

class CategorySelections 
  extends React.Component<
    CategorySelectionsAttrs, { selected: boolean[] }
  > {

  constructor(props) {
    super(props);
    this.state = { selected: [] };
  }

  setSelection(i: number, val: boolean) {
    this.setState(state => {
      state.selected[i] = val;
      return state;
    })
  }

  createCategoryItem(i: number) : JSX.Element {
    const tag = this.props.categories[i];
    let isHighlighted = this.props.activeCategories[tag._id];

    return (
      <CategoryView
        key={`cat-item-${i}`}
        isAdmin={this.props.isAdmin}
        tagName={tag}
        selected={this.props.applyToImageMode ? false : this.state.selected[i]}
        highlighted={isHighlighted}
        color={this.props.categories[i].color}
        applyToImageMode={this.props.applyToImageMode}

        select={() => {
          if (!this.state.selected[i]) {
            this.props.unshiftSelection(i);
            for (let j = this.state.selected.length-1; j >= 1; j--)
              this.setSelection(j, this.state.selected[j-1]);
            this.setSelection(0, true);
          } else {
            this.setSelection(i, false);
          }
        }}

        applyToImage={() => {
          this.props.applyToImage(this.props.categories[i]);
        }}

        activatePopup={() => {
          console.log('activating edit tag popup with', tag);
          this.props.store.activatePopup('editTag', {
            projectId: this.props.projectId,
            uid: this.props.uid,
            tag,

            updateTag: this.props.updateTag,
            removeTag: this.props.removeTag,
          });
        }}
      />
    );
  }

  render() {
    /*if (this.state.selected.length != this.props.categories.length)
      this.setState({ 
        selected: initArr<boolean>(this.props.categories.length, false) 
      });*/
    
    return (
      <div className='category-container'>
        {this.props.applyToImageMode && 
          <span className='category-list-header'>Click Tags to Apply to Images</span>}
        
        <div className='category-content'>{
          (this.props.categories || [])
            .map((_, i) => 
              this.createCategoryItem(i)
            )
            .concat((this.props.applyToImageMode || !this.props.isAdmin) ? 
              [] : (
                <div key='fuck' className='category-item new-category-button selected'
                  onClick={this.props.addTag}
                >
                  <span className='category-item-new-icon' />
                  <span className='category-item-title'>Add Tag</span>
                </div>
              )
            )
        }</div>
      </div>
    );
  }
}

export default withPopupStore(CategorySelections);